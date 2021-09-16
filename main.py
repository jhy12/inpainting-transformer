import os
import cv2
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms as T
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import matplotlib.pyplot as plt

from dataset import MVTecAD_loader
import argparse
from ast import literal_eval

from model import InTra

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default='../mvtec_anomaly_detection/bottle/', type=str)

    parser.add_argument('--image_size', default='(256,256)', type=str)
    parser.add_argument('--num_epochs', default=2000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--train_ratio', default=0.9, type=float)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--pretrained', default=None, type=str)

    parser.add_argument('--ckpt', default='./ckpt/InTra/MVTAD_bottle/', type=str)
    parser.add_argument('--is_infer', action='store_true')

    return parser.parse_args()

def train(model, train_loader):
    model.train()
    train_loss = 0
    for data, label in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = model._process_one_batch(data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    train_loss /= len(train_loader.dataset)
    return train_loss

def valid(model, valid_loader):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for data, label in valid_loader:
            data = data.to(device)
            #loss, image_recon, image_reassembled, msgms_map =  model._process_one_image(data)
            loss, _, _, _ =  model._process_one_image(data)
            valid_loss += loss.item()
    valid_loss /= len(valid_loader.dataset)
            
    return valid_loss

def tensor2nparr(tensor):
    np_arr = tensor.detach().cpu().numpy()
    np_arr = (np.moveaxis(np_arr, 1, 3)*255).astype(np.uint8)
    return np_arr

if __name__ =='__main__':
    global args
    args = parse_args()
    args.image_size = literal_eval(args.image_size)
    args.results = os.path.join(args.ckpt, 'results')
    #args.prefix = os.path.basename(os.path.dirname(args.test_dir))
    #print(args.prefix)

    train_loader, valid_loader = MVTecAD_loader(args.image_dir, args.image_size, args.train_ratio, args.batch_size, num_workers=4, is_inference=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    seed = 42
    if not os.path.exists(args.ckpt):
        os.mkdir(args.ckpt)
    if not os.path.exists(args.results):
        os.mkdir(args.results)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = InTra(grid_size_max=int(args.image_size[0]/16)).to(device)

    is_train = not args.is_infer
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
    
    best_loss = 1e+15
    valid_loss = 1e+15
    if is_train:
        if args.pretrained:
            model_state_dict = torch.load(os.path.join(args.pretrained, 'model_best.nn'), map_location='cuda')
            model.load_state_dict(model_state_dict)
        num_epochs = args.num_epochs
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader)
            if epoch % 100 == 0:
                valid_loss = valid(model, valid_loader)
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(args.ckpt, 'model_best.nn'))
            torch.save(model.state_dict(), os.path.join(args.ckpt, 'model_last.nn'))
            print('epoch [{}/{}], train loss: {:.6f}, valid_loss: {:.6f}, best_loss: {:.6f}'.format(epoch + 1, num_epochs, train_loss, valid_loss, best_loss))
    else:
        model_state_dict = torch.load(os.path.join(args.ckpt, 'model_last.nn'), map_location='cuda')
        model.load_state_dict(model_state_dict)

        test_loader = MVTecAD_loader(args.image_dir, args.image_size, args.train_ratio, 1, num_workers=0, is_inference=True)
        model.eval()
        test_loss = 0

        with torch.no_grad():
            for data, label in test_loader:
                data = data.to(device)
                loss, image_recon, image_reassembled, msgms_map =  model._process_one_image(data)
                test_loss += loss.item()

                data_arr = tensor2nparr(data)
                image_recon_arr = tensor2nparr(image_recon)
                image_reassembled_arr = tensor2nparr(image_reassembled)
                msgms_map_arr = tensor2nparr(msgms_map)

                cv2.imshow('image', data_arr[0])
                cv2.imshow('image_recon_arr', image_recon_arr[0])
                cv2.imshow('msgms_map_arr', msgms_map_arr[0])
                cv2.imshow('heatmap', cv2.applyColorMap(msgms_map_arr[0], cv2.COLORMAP_JET))
                cv2.waitKey(0)
        
        test_loss /= len(test_loader.dataset)