# -*- coding:utf-8 -*-
# 作者：KKKC
import math
import os
import numpy
import torch
import torch.nn as nn
from PIL import Image
from model import Generator
from config import *
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim


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
def plot_example(low_res_folder, gen):
    files = os.listdir(low_res_folder)
    # gen.eval()
    for file in files:
        image = Image.open(low_res_folder + file)
        image = Tensor_transform(image)
        image = torch.unsqueeze(image, dim=0)  # 需要重新赋值一下
        # with torch.no_grad():
        SR_img = gen(image)
        SR_img = torch.squeeze(SR_img, dim=0)
        SR_img = PIL_transform(SR_img)
        # print(SR_img.shape)
        # save_image(SR_img,f"save_result/{file}")
        SR_img.save("./save_result/{}".format(file))


# 保存结果
def save_image(low_res_folder, gen):
    files = os.listdir(low_res_folder)
    gen.eval()
    for file in files:
        image = Image.open(low_res_folder + file)
        image = Tensor_transform(image)
        image = torch.unsqueeze(image, dim=0)
#       print(image.shape)
        if DEVICE == "cuda":
            image = image.type(torch.cuda.FloatTensor) # 在colab中防止报错
        with torch.no_grad():
            SR_img = gen(image)
            SR_img = torch.squeeze(SR_img, dim=0)  # 去掉Batch
            SR_img = PIL_transform(SR_img)
            SR_img.save(f"save_result/{file}")
            
# 
def save_test_image(low_res_folder, gen):
    files = os.listdir(low_res_folder)
    gen.eval()
    for file in files:
        path = low_res_folder + file
        print(path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255.
        img = torch.from_numpy(numpy.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(config.DEVICE)

        with torch.no_grad():
            output = gen(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = numpy.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        cv2.imwrite('save_result/{}'.format(file), output)

# 计算psnr指标，PSNR = 10 * log10(Max(I**2) / MSE)
def caculate_psnr(hr_image, sr_image):
    mse = numpy.mean((hr_image / 255. - sr_image / 255.) ** 2)  # 如果输入的是彩色图像，则mean操做会对三个通道进行平均，不用再/3
    if mse < 1.0e-10:
        return 100  # 如果两图片差距很小代表完美组合
    psnr = 20 * math.log10(1.0 / math.sqrt(mse))
    return psnr


# 计算ssim指标
def caculate_ssim(hr_image, sr_image):
    return ssim(hr_image, sr_image, multichannel=True)  # 多通道，彩色图像


# 所有图片的PSNR平均值
def M_psnr(hr_folder, sr_folder):
    psnr_sum = []
    hr_list = []
    for img in os.listdir(hr_folder):
        if img.endswith(".png"):
            hr_list.append(img)
    # sr_list = os.listdir(sr_folder)
    for img in hr_list:
        hr_img = Image.open(os.path.join(hr_folder, img))
        sr_img = Image.open(os.path.join(sr_folder, img))
        hr_img = numpy.array(hr_img)
        sr_img = numpy.array(sr_img)
        psnr = caculate_psnr(hr_img, sr_img)
        psnr_sum.append(psnr)
    return numpy.mean(psnr_sum)


# 所有图像的SSIM平均值
def M_ssim(hr_folder, sr_folder):
    ssim_num = []
    hr_list = []
    for img in os.listdir(hr_folder):
        if img.endswith(".png"):
            hr_list.append(img)
    # sr_list = os.listdir(sr_folder)
    for img in hr_list:
        hr_img = Image.open(os.path.join(hr_folder, img))
        sr_img = Image.open(os.path.join(sr_folder, img))
        hr_img = numpy.array(hr_img)
        sr_img = numpy.array(sr_img)
        ssim = caculate_ssim(hr_img, sr_img)
        ssim_num.append(ssim)
    return numpy.mean(ssim_num)


# if __name__ == "__main__":
#     Gen = Generator()
#     Gen.load_state_dict(torch.load("./save_model/net_G_352_26.336534.pth", map_location="cpu"))
#     save_image(low_res_folder="./test_image/", gen=Gen)
#     hr_folder = HR_DIR
#     sr_folder = SR_DIR
#     mean_psnr = M_psnr(hr_folder, sr_folder)
#     mean_ssim = M_ssim(hr_folder, sr_folder)
#     print("{:.4f}".format(mean_psnr))
#     print("{:.4f}".format(mean_ssim))
