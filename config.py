# -*- coding: utf-8 -*-
# @Time : 2022/10/23 21:17
# @Author : KKKc
# @FileName: config.py

import torch
import torch.nn as nn

import torch
from PIL import Image
from torchvision import transforms
# import albumentations as A  # 图像增强的库
# from albumentations.pytorch import ToTensorV2

LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN = "gen.pth"
CHECKPOINT_DISC = "disc.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARING_RATE = 1e-4
NUM_EPOCHS = 2000
BATCH_SIZE = 16
NUM_WORKERS = 4
HIGH_RES = 96
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 3

# HR图像的transform
H_transform = transforms.Compose(
    [
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
        transforms.ToTensor(),
    ]
)

# IR图像的transform
I_transform = transforms.Compose(
    [
        transforms.Resize((LOW_RES,LOW_RES),interpolation=Image.BICUBIC),
        transforms.Normalize(mean=[0,0,0],std=[1,1,1]),
        transforms.ToTensor()
    ]
)

# all的transform
all_tranform = transforms.Compose(
    [
        transforms.RandomCrop((HIGH_RES,HIGH_RES)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=0.5),
    ]
)