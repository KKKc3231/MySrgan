# -*- coding:utf-8 -*-
# 作者：KKKC

import os
import torch
import torch.nn as nn
from PIL import Image
from model import Generator
from config import *
from torchvision.utils import save_image

# 梯度惩罚
def gradient_penalty(critic, real, fake, device=None):
    # 获取信息
    batch_size, c, h, w = real.shape
    print(batch_size)
    alpha = torch.rand((batch_size, 1, 1, 1)).repeat(1, c, h, w).to(device)
    mixed_image = alpha * real + (1 - alpha) * fake
    mixed_score = critic(mixed_image)
    # 计算梯度
    gradient = torch.autograd.grad(
        inputs=mixed_image,
        outputs=mixed_score,
        grad_outputs=torch.ones_like(mixed_score),
        create_graph=True,
        retain_graph=True,
    )[0]
    gredient = gradient.view(gradient.shape[0], -1)
    # 计算二范数
    gradient_norm = gredient.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)  # 梯度损失
    return gp

# 绘制图片
def plot_example(low_res_folder,gen):
    files = os.listdir(low_res_folder)
    # gen.eval()
    for file in files:
        image = Image.open(low_res_folder + file)
        image = Tensor_transform(image)
        image = torch.unsqueeze(image,dim=0)  # 需要重新赋值一下
        # with torch.no_grad():
        SR_img = gen(image)
        SR_img = torch.squeeze(SR_img, dim=0)
        SR_img = PIL_transform(SR_img)
        # print(SR_img.shape)
        # save_image(SR_img,f"save_result/{file}")
        SR_img.save("./save_result/{}".format(file))

# 保存结果
def save_image(low_res_folder,gen):
    files = os.listdir(low_res_folder)
    gen.eval()
    for file in files:
        image = Image.open(low_res_folder + file)
        image = Tensor_transform(image)
        image = torch.unsqueeze(image,dim=0)
        print(image.shape)
        with torch.no_grad():
            SR_img = gen(image)
            SR_img = torch.squeeze(SR_img,dim=0) # 去掉Batch
            SR_img = PIL_transform(SR_img)
            SR_img.save(f"save_result/{file}")


if __name__ == "__main__":
    Gen = Generator()
    Gen.load_state_dict(torch.load("./save_model/net_G_1995.pth",map_location="cpu"))
    plot_example(low_res_folder="./test_image/",gen=Gen)
