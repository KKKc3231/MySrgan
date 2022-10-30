# -*- coding: utf-8 -*-
# @Time : 2022/10/23 21:12
# @Author : KKKc
# @FileName: VGG_loss.py
import torch
import config
import torch.nn as nn
from torchvision.models import vgg19
from config import *


# VGG损失
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(DEVICE)
        self.loss = nn.MSELoss()
        # 不更新VGG的梯度，VGG不学习
        for parameters in self.vgg.parameters():
            parameters.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)  # input是Groud True图像
        vgg_target_features = self.vgg(target)  # target是放大后的IR
        loss = self.loss(vgg_input_features, vgg_target_features)
        return loss




