import os
import PIL
from PIL import Image
import numpy as np
import cv2
import random

import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision import transforms

def get_image_list(path):
    image_list = []
    for (root, dirs, files) in os.walk(path):
        for fname in files:
            if os.path.splitext(os.path.basename(fname))[-1] in ['.jpeg', '.jpg', '.bmp', '.png', '.tif']:
                image_list.append(os.path.join(root, fname))
    return image_list

def make_test_label(test_imgdir, test_labdir, test_imgpath, img_size):
    # test_imgdir : bottle/test
    # test_labdir : bottle/ground_truth
    # test_imgpath : bottle/test/broken_large/000.png ||| bottle/test/good/000.png
    # test_labpath : bottle/ground_truth/broken_large/000_mask.png
    test_labpath_base = test_imgpath.replace(test_imgdir, test_labdir)
    test_labpath = os.path.join(test_labpath_base.split('.png')[0] + '_mask.png')
    if os.path.exists(test_labpath):
        label = cv2.resize(cv2.imread(test_labpath, cv2.IMREAD_GRAYSCALE), img_size, interpolation=cv2.INTER_NEAREST)
        label[label>=1] = 1
        return (label, 1)
    else:
        print('label not exists: ', test_imgpath)
        return (np.zeros(shape=img_size) , 0)


class MVTecAD(data.Dataset):

    def __init__(self, image_list, label_list, transform):
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.image_list[index])
        label = self.label_list[index]
        return self.transform(image), label

    def __len__(self):
        return len(self.image_list)


def MVTecAD_loader(image_dir, image_size, train_ratio=0.9, batch_size=1, num_workers=0, is_inference=False, seed=1234):
    assert train_ratio >=0.0 and train_ratio <=1.0
    # automatically returns two dataloaders. train & valid depends on train_ratio
    random.seed(seed)
    transform_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    transform_infer = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    train_imgdir = os.path.join(image_dir, os.path.join('train', 'good'))
    test_imgdir = os.path.join(image_dir, 'test')
    test_labdir = os.path.join(image_dir, 'ground_truth')

    train_image_list = get_image_list(train_imgdir)
    random.shuffle(train_image_list)
    train_image_list, valid_image_list = train_image_list[:int(len(train_image_list)*train_ratio)], train_image_list[int(len(train_image_list)*train_ratio):]
    train_label_list = [(np.zeros(image_size, dtype=np.uint8), 0)]*len(train_image_list)
    valid_label_list = [(np.zeros(image_size, dtype=np.uint8), 0)]*len(valid_image_list)

    # test dataset include segmentation labels - make segmentation labels for normal samples
    test_image_list = get_image_list(test_imgdir)
    test_label_list = [make_test_label(test_imgdir, test_labdir, x, image_size) for x in test_image_list]

    if not is_inference:
        train_dataset = MVTecAD(train_image_list, train_label_list, transform_train)
        train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        if train_ratio < 1.0:
            valid_dataset = MVTecAD(valid_image_list, valid_label_list, transform_infer)
            valid_dataloader = data.DataLoader(dataset=valid_dataset, batch_size=1, num_workers=0, shuffle=False)
        else:
            valid_dataloader = None
        return train_dataloader, valid_dataloader
    else:
        infer_dataset = MVTecAD(test_image_list, test_label_list, transform_infer)
        infer_dataloader = data.DataLoader(dataset=infer_dataset, batch_size=1, num_workers=0, shuffle=False)
        return infer_dataloader

def imshow(x_0):
    for i in range(list(x_0.size())[0]):
        img = x_0[i].detach().cpu().numpy()
        img = img*255.0
        img = img.astype(np.uint8)
        img = np.moveaxis(img, 0, 2)
        cv2.imshow('img', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    image_dir = '../mvtec_anomaly_detection/bottle/'
    image_size = (256, 256)
    data_loader, _ = MVTecAD_loader(image_dir, image_size, 1.0, 10, 0, is_inference=False, seed=1234)
    x_0, _ = iter(data_loader).next()
    imshow(x_0)