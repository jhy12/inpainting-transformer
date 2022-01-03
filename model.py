import torch
from torch import nn
from ssim import SSIMLoss
from msgms import MSGMSLoss
import torch.nn.functional as F
import copy, math


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        if self.dropout is not None:
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            return x + sublayer(self.norm(x))
        
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class InTraEncoder(nn.Module):
    def __init__(self, layer, N):
        super(InTraEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
    
    def forward(self, x, mask=None):
        # src_mask : meaningless tokens..(pad token)
        layers = []
        for i, layer in enumerate(self.layers):
            if i > len(self.layers)/2:
                x = x + layers[len(self.layers)-i]
            
            x = layer(x, mask)
            layers.append(x)
        return self.norm(x)

class InTraEncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(InTraEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    # [B, nh, seq_len, d_k]
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiheadedFeatureSelfAttention(nn.Module):
    def __init__(self, h, d_model, dropout=None):
        super(MultiheadedFeatureSelfAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.MLP_q = nn.Sequential(*[nn.Linear(d_model, d_model * 2), 
                                    nn.GELU(),
                                    nn.Linear(d_model*2, int(d_model/2))])
        self.MLP_k = nn.Sequential(*[nn.Linear(d_model, d_model * 2), 
                                    nn.GELU(),
                                    nn.Linear(d_model*2, int(d_model/2))])
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

        self.attn = None
        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # query, key : [B, h, seq_len, d_k//2]
        # value : [B, h, seq_len, d_k]
        query = self.MLP_q(query).view(nbatches, -1, self.h, self.d_k // 2).transpose(1, 2)
        key = self.MLP_k(key).view(nbatches, -1, self.h, self.d_k // 2).transpose(1, 2)
        value = self.linear_v(value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        # [B, h, seq_len, d_k//2] * [B, h, d_k//2, seq_len] = [B, h, seq_q, seq_k]
        # [B, h, seq_q, seq_k] * [B, h, seq_v, d_k] = [B, h, seq_q, d_k]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linear(x)


from einops import rearrange

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if self.dropout is not None:
            return self.w_2(self.dropout(F.gelu(self.w_1(x))))
        else:
            return self.w_2(F.gelu(self.w_1(x)))

class Generator(nn.Module):
    def __init__(self, d_model, n_pixels):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, n_pixels)
    
    def forward(self, x):
        return self.proj(x)

def g(c, L=7):
    return max(1, c-int(L/2))

class InTra(nn.Module):
    def __init__(self, grid_size_max=16, K=16, L=7, C=3, N=13, h=8, max_len=1024, d_model=512):
        super(InTra, self).__init__()
        # img : H x W x C
        # M = H / K, N = W / K
        # K : predefined patch size(following ViT, K=16)
        # L : length of square subgrids(L=7)
        self.K = K
        self.C = C

        self.n_pixels = K*K*C
        self.lin_proj = nn.Linear(K*K*C, d_model)
        self.x_inpaint = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding_loc = nn.Parameter(torch.randn(1, L*L, d_model))
        self.pos_embedding_glb = nn.Parameter(torch.randn(1, grid_size_max*grid_size_max, d_model))
        self.self_attn = MultiheadedFeatureSelfAttention(h, d_model, dropout=None)
        self.feed_forward = PositionwiseFeedForward(d_model, d_model*4)
        self.encoder_layer = InTraEncoderLayer(size=d_model, self_attn=self.self_attn, feed_forward=self.feed_forward,dropout=None)
        self.encoder = InTraEncoder(self.encoder_layer, N)
        self.generator = Generator(d_model, self.n_pixels)

        self.mse = torch.nn.MSELoss(reduction='mean')
        self.msgms = MSGMSLoss(num_scales=3, in_channels=3)
        self.ssim = SSIMLoss(kernel_size=11, sigma=1.5)

    def forward(self, x):
        encoded = self.encoder(x)
        encoded_avg = torch.mean(encoded, dim=1)
        generated_patch = self.generator(encoded_avg)
        return generated_patch

    # Training sequence
    # image : M, N(not N, M... differnet from original paper)
    # choose grid position (r, s)
    # choose grid position (t, u) to inpaint

    def _process_one_batch(self, batch_img):
        x, gt = self._preprocess_train_batch(batch_img, self.x_inpaint, self.pos_embedding_glb, self.lin_proj)
        recon = self.forward(x)
        gt = gt.reshape(gt.size(0), self.C, self.K, self.K)
        recon = recon.reshape(recon.size(0), self.C, self.K, self.K)
        loss, _, _ = self._compute_loss(recon, gt)
        return loss

    def _preprocess_train_batch(self, batch_img, x_inpaint, pos_embedding_glb, lin_proj, K=16, L=7):
        # get img size
        B, C, H, W = batch_img.size()
        # confusing notations.. we use M x N not N x M for original papers.
        M = int(H / K)
        N = int(W / K)

        # We start from 1~M*N and -1 at the pos_embedding_glb_idx -= 1 to make the range 0~M*N-1
        pos_embedding_glb_grid = torch.arange(1, N*M+1, dtype=torch.long).reshape(M, N)

        # sampled_rs_idx : [B, 2] / sampled_rs_idx[b] = [r, s] (0 <= r <= M-L / 0 <= s <= N-L)
        sampled_rs_idx = torch.cat([torch.randint(0, M-L+1, (B,1)), torch.randint(0, N-L+1, (B,1))], dim=1)
        # pos_embedding_glb_idx : [B, L*L] (size : L*L, but 0 <= value < M*N)
        # sampled subgrid's positional embedding index : i, j(sampled_rs_idx) -> i*N + j(pos_embedding_glb_grid)
        pos_embedding_glb_idx = torch.vstack([pos_embedding_glb_grid[r:r+L, s:s+L].unsqueeze(0) for r, s in sampled_rs_idx]).unsqueeze(dim=1)
        pos_embedding_glb_idx = pos_embedding_glb_idx.reshape(pos_embedding_glb_idx.size(0), -1)
        pos_embedding_glb_idx -= 1

        # sampled subgrid's values...
        batch_subgrid = torch.vstack([batch_img[l,:,K*r:K*(r+L),K*s:K*(s+L)].unsqueeze(0) for l, (r, s) in enumerate(sampled_rs_idx)])

        # batch subgrid : [B, C, L*K, L*K] / corresponding positional embedding index(pos_embedding_glb_idx) : [B, L, L] (value range : 1~L**2)
        # now... convert to transformer input formats..
        # pos_embedding_glb_idx : [B, L*L]
        # pos_embedding_glb : [1, M*N, d_model]
        # pos_embedding : [B, L*L, d_model]
        pos_embedding = torch.zeros((B, L*L, pos_embedding_glb.size(2))).to(pos_embedding_glb.device)
        for b in range(B):
            for n in range(pos_embedding_glb_idx.size(1)):
                pos_embedding[b,n,:] = pos_embedding_glb[:,pos_embedding_glb_idx[b, n],:]
                
        #pos_embedding = pos_embedding_glb[pos_embedding_glb_idx]
        # h, w : L
        batch_subgrid_flatten = rearrange(batch_subgrid, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=K, p2=K)

        # inpaint index
        # sampled_tu_idx : [B]
        # it extracts from the flattened array with hidden dimension 
        sampled_tu_idx = torch.randint(0, L*L, (B,))
        sampled_tu_idx_one_hot = F.one_hot(sampled_tu_idx, L*L)
        sampled_tu_idx_T = sampled_tu_idx_one_hot.bool()
        sampled_tu_idx_F = torch.logical_not(sampled_tu_idx_T)

        batch_subgrid_inpaint = batch_subgrid_flatten[sampled_tu_idx_T]
        pos_embedding_inpaint = pos_embedding[sampled_tu_idx_T].unsqueeze(1)

        batch_subgrid_emb_input = batch_subgrid_flatten[sampled_tu_idx_F].reshape(B, L*L-1, K*K*C)
        pos_embedding_emb_input = pos_embedding[sampled_tu_idx_F].reshape(B, L*L-1, -1)

        # concat at seq dimension
        batch_subgrid_input = torch.cat([x_inpaint + pos_embedding_inpaint, lin_proj(batch_subgrid_emb_input) + pos_embedding_emb_input], dim=1)
    
        return batch_subgrid_input, batch_subgrid_inpaint

    def _compute_loss(self, patch_recon, patch_gt):
        mse_loss = self.mse(patch_gt, patch_recon)
        msgms_loss, msgms_map = self.msgms(patch_gt, patch_recon)
        ssim_loss, ssim_map = self.ssim(patch_gt, patch_recon)
        #total_loss = mse_loss + 0.01 * msgms_loss + 0.01 * ssim_loss 
        total_loss = mse_loss + msgms_loss + ssim_loss 
        return total_loss, msgms_map, ssim_map

   
    def _process_one_image(self, image, K=16, L=7, infer_batch_size=1):
        image_recon, gt, loss = self._process_infer_image(image, K, L)
        '''
        if infer_batch_size > 1:
            image_recon, gt, loss = self._process_infer_image(image, K, L)
        else:
            image_recon, gt, loss = self._process_infer_image_batch(image, infer_batch_size, K, L)
        '''
        _, msgms_map, _ = self._compute_loss(image_recon, gt)
        return loss, image_recon, gt, msgms_map 
    
    def _process_infer_image_batch(self, image, infer_batch_size=64, K=16, L=7):
        patches_recon = []
        patches_gt = []
        patches_loss = []
        # get img size
        B, C, H, W = image.size()
        assert B == 1
        # confusing notations.. we use M x N not N x M for original papers.
        M = int(H / K)
        N = int(W / K)

        subgrid_input_batch = []
        for t in range(M):
            for u in range(N):
                r = g(t, L) - max(0, g(t, L) + L - M - 1)
                s = g(u, L) - max(0, g(u, L) + L - N - 1)
                subgrid_input, subgrid_inpaint = self._process_subgrid(image, self.x_inpaint, self.pos_embedding_glb, self.lin_proj, M, N, r, s, t, u, K, L)

                if len(subgrid_input_batch) == infer_batch_size or ((t == M-1) and (u == N-1)):
                    # subgrid_input_batch : [infer_batch_size, L*L, d_model]
                    subgrid_input_batch = torch.vstack(subgrid_input_batch)
                    patch_recon_batch = self.forward(subgrid_input_batch)


                if len(subgrid_input_batch) < infer_batch_size:
                    subgrid_input_batch.append(subgrid_input)

                patch_recon = self.forward(subgrid_input)
                patches_recon.append(patch_recon)
                patches_gt.append(subgrid_inpaint)

                p_recon_r = patch_recon.reshape(patch_recon.size(0), self.C, self.K, self.K)
                p_gt_r = subgrid_inpaint.reshape(subgrid_inpaint.size(0), self.C, self.K, self.K)
                patches_loss.append(self._compute_loss(p_recon_r, p_gt_r)[0])

        image_recon = self._combine_recon_patches(patches_recon, M, N, K)
        gt = self._combine_recon_patches(patches_gt, M, N, K)
        loss = torch.mean(torch.tensor(patches_loss)) / B
        return image_recon, gt, loss

    def _process_infer_image(self, image, K=16, L=7):
        patches_recon = []
        patches_gt = []
        patches_loss = []
        # get img size
        B, C, H, W = image.size()
        assert B == 1
        # confusing notations.. we use M x N not N x M for original papers.
        M = int(H / K)
        N = int(W / K)

        for t in range(M):
            for u in range(N):
                r = g(t, L) - max(0, g(t, L) + L - M - 1)
                s = g(u, L) - max(0, g(u, L) + L - N - 1)
                subgrid_input, subgrid_inpaint = self._process_subgrid(image, self.x_inpaint, self.pos_embedding_glb, self.lin_proj, M, N, r, s, t, u, K, L)
                patch_recon = self.forward(subgrid_input)
                patches_recon.append(patch_recon)
                patches_gt.append(subgrid_inpaint)

                p_recon_r = patch_recon.reshape(patch_recon.size(0), self.C, self.K, self.K)
                p_gt_r = subgrid_inpaint.reshape(subgrid_inpaint.size(0), self.C, self.K, self.K)
                patches_loss.append(self._compute_loss(p_recon_r, p_gt_r)[0])

        image_recon = self._combine_recon_patches(patches_recon, M, N, K)
        gt = self._combine_recon_patches(patches_gt, M, N, K)
        loss = torch.mean(torch.tensor(patches_loss)) / B
        return image_recon, gt, loss

    def _process_subgrid(self, image, x_inpaint, pos_embedding_glb, lin_proj, M, N, r, s, t, u, K, L):
        # change r, s range from 1 <= r, s <= M-L+1, N-L+1
        #                     to 0 <= r, s <= M-L, N-L
        r = min(max(0, r-1), M-L)
        s = min(max(0, s-1), N-L)
        B, C, H, W =image.size()
        # subgrid -> [1, C, K*L, K*L]
        subgrid = image[:,:,K*r:K*(r+L),K*s:K*(s+L)]
        # subgrid_flatten : [1, L*L, K*K*C]
        subgrid_flatten = rearrange(subgrid, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=K, p2=K)

        # pos_embedding_glb_idx : [1, L*L]
        pos_embedding_glb_grid = torch.arange(1, M*N+1, dtype=torch.long).reshape(M, N)
        pos_embedding_glb_idx = pos_embedding_glb_grid[r:r+L, s:s+L].unsqueeze(0)
        pos_embedding_glb_idx = pos_embedding_glb_idx.reshape(pos_embedding_glb_idx.size(0), -1)
        pos_embedding_glb_idx -= 1

        # pos_embedding_grid : [1, L*L, d_model]
        pos_embedding = torch.zeros(1, L*L, pos_embedding_glb.size(2)).to(pos_embedding_glb.device)
        for n in range(pos_embedding_glb_idx.size(1)):
            pos_embedding[:,n,:] = pos_embedding_glb[:,pos_embedding_glb_idx[:,n],:]

        # r, s, t, u ... M x N
        # t, u : 0 <= t <= M / 0 <= u <= n

        # tu_1d_idx : 0 <= val < M*N
        # but it should be shape in L*L
        tu_1d_idx = torch.tensor([(t-r) * L + (u-s)], dtype=torch.long)
        tu_one_hot = F.one_hot(tu_1d_idx, L*L)
        tu_idx_T = tu_one_hot.bool()
        tu_idx_F = torch.logical_not(tu_idx_T)

        subgrid_inpaint = subgrid_flatten[tu_idx_T]
        pos_embedding_inpaint = pos_embedding[tu_idx_T].unsqueeze(1)

        subgrid_emb_input = subgrid_flatten[tu_idx_F].reshape(B, L*L-1, K*K*C)
        pos_embedding_emb_input = pos_embedding[tu_idx_F].reshape(B, L*L-1, -1)

        subgrid_input = torch.cat([x_inpaint + pos_embedding_inpaint, lin_proj(subgrid_emb_input) + pos_embedding_emb_input], dim=1)
        return subgrid_input, subgrid_inpaint

    def _combine_recon_patches(self, patch_list, M, N, K):
        # patch_list : list of M*N [1, K*K*C] tensor
        # patches_concat : [M*N, 1, K*K*C]
        patch_list = [x.unsqueeze(0) for x in patch_list]
        patches_concat = torch.cat(patch_list, dim=0)
        # patches_concat : [1, M*N, K*K*C]
        patches_concat = patches_concat.permute(1, 0, 2)
        # recon_image : [1, C, H, W]
        recon_image = rearrange(patches_concat, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2) ', h = M, w = N, p1= K, p2=K)
        return recon_image


if __name__ == "__main__":
    model = InTra()

    # train sequence
    batch_img = torch.randn(4, 3, 256, 256)
    loss = model._process_one_batch(batch_img)

    # test sequence
    image = torch.randn(1, 3, 256, 256)
    recon, gt = model._process_one_image(image)

