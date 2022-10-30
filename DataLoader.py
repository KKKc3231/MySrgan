# -*- coding: utf-8 -*-
# @Time : 2022/10/24 10:14
# @Author : KKKc
# @FileName: Data.py
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import os
from PIL import Image
from torchvision import transforms

path = "E:/SRGAN-SRCNN/Data/SR_training_datasets/BSDS200/" # 改成自己的数据集地址即可

# 图像处理操作，包括随机裁剪，转换张量，且不需要两次ToTensor()
H_transform = transforms.Compose(
    [
        transforms.RandomCrop(96),
        transforms.ToTensor()
    ]
)

L_transform = transforms.Compose(
    [
        transforms.Resize((24,24)), # 改成默认为BICUBIC了
        transforms.Normalize(mean=[0,0,0],std=[1,1,1]),
    ]
)

class MyDataset(Dataset):
    def __init__(self,path=path,H_transforms=H_transform,L_transforms=L_transform,ex=10):
        self.H_transform = H_transforms
        self.L_transform = L_transforms
        # 200张图片的路径
        files = os.listdir(path)
        self.img = [path + file for file in files]
        # np.random.shuffle(self.img) # 随机的打乱

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        HR_img_path = self.img[index]
        HR_image = Image.open(HR_img_path)
        HR_img = self.H_transform(HR_image)
        LR_img = self.L_transform(HR_img)
        return LR_img,HR_img
