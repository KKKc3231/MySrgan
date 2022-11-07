# -*- coding: utf-8 -*-
# @Time : 2022/10/24 15:49
# @Author : KKKc
# @FileName: train.py
import torch
import torch.optim as optim
from model import *
from config import *
from VGG_loss import *
from Data import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import gradient_penalty

# Dataset
path = "E:/SRGAN-SRCNN/Data/SR_training_datasets/BSDS200/"
Dataset = MyDataset(path = "E:/SRGAN-SRCNN/Data/SR_training_datasets/BSDS200/")
Dataloader = DataLoader(dataset=Dataset,batch_size=1,shuffle=True)
L_GP = 10 # 梯度惩罚系数

# best_psnr和best_ssim初始化
save_image(low_res_folder="./test_image/", gen=net_G)
best_psnr = M_psnr(HR_DIR,SR_DIR)
best_ssim = M_ssim(HR_DIR,SR_DIR)
print("best_psnr:{}".format(best_psnr))
print("best_ssim:{}".format(best_ssim))

# 搭建模型
net_D = Discriminator()
net_G = Generator()
net_D.to(DEVICE)
net_G.to(DEVICE)

# D和G的优化器
optimizer_D = optim.Adam(net_D.parameters(),lr=1e-4)
optimizer_G = optim.Adam(net_G.parameters(),lr=1e-4)

# 损失函数
loss_mse = nn.MSELoss()  # mse l2损失
loss_bce = nn.BCEWithLogitsLoss()  # 交叉熵损失
VGG_loss = VGGLoss()  # vgg_loss

# train
for epoch in range(NUM_EPOCHS):
    net_D.train() # train
    net_G.train()
    processBar = tqdm(enumerate(Dataloader))  # 进度条显示
    for i,(L_img,H_img) in processBar:
        L_img, H_img = L_img.to(DEVICE),H_img.to(DEVICE)
        FakeImg = net_G(L_img).to(DEVICE)
        print(FakeImg.shape)
        # Discriminator loss
        optimizer_D.zero_grad()
        real_out = net_D(H_img)
        fake_out = net_D(FakeImg)
        # disc_loss_real为判别器对HR图像的损失
        # 对于GAN的损失这一块的代码，一直没搞懂。现在来看，我的理解是对于判别器：
        # 目标是 max log(D(x)) + log(1 - D(G(z)) --> min -(log(D(x)) + log(1 - D(G(z)) --> 这是交叉熵
        disc_loss_real = loss_bce(real_out,torch.ones_like(real_out) - 0.1*torch.rand_like(real_out)) # 标签平滑
        # disc_loss_fake为判别器对LR图像的损失
        disc_loss_fake = loss_bce(fake_out,torch.zeros_like(fake_out))
        # GP正则
        gp = gradient_penalty(net_D,H_img,FakeImg,device=DEVICE)
        dLoss = disc_loss_real + disc_loss_fake + L_GP * gp
        dLoss.backward(retain_graph=True)
        optimizer_D.step()

        # Generator loss
        # 对于G来说，要min log(1 - D(G(z)) --> -log(D(G(z)) ??
        optimizer_G.zero_grad()
        gLoss_SR = loss_mse(FakeImg,H_img)  # 像素点的mse损失
        gLoss_GAN = 0.001 * (loss_bce(FakeImg,torch.ones_like(FakeImg)))
        # gLoss_GAN = 0.001 * (torch.mean(1.0 - fake_out))  # 生成器的对抗损失
        gLoss_VGG = 0.006 * VGG_loss(FakeImg,H_img)  # LR和HR经过Generator后的特征图的损失
        gLoss = gLoss_SR + gLoss_VGG  # 总的损失
        gLoss.backward()
        optimizer_G.step()

        # tqdm进度条可视化，当batch_size > 1 的时候会报错
#         processBar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
#             epoch, NUM_EPOCHS, dLoss.item(), gLoss.item(), real_out.item(), fake_out.item()))
        print("-----------------------Epoch：{}----------------------".format(epoch))
    
    save_image(low_res_folder="/content/drive/MyDrive/SRGAN/test_image/", gen=net_G)
    # 计算平均的PSNR和SSIM
    m_psnr = M_psnr(HR_DIR,SR_DIR)
    m_ssim = M_ssim(HR_DIR,SR_DIR)
    if m_psnr > best_psnr:  # 以psnr指标为判别标准，只保存有提高的模型
      GREEN = '\033[92m'
      END_COLOR = '\033[0m'
      print(GREEN + "New_Best_PSNR:{} @epoch:{}".format(m_psnr,epoch) + END_COLOR)  # 带颜色的打印一下，明显 
      torch.save(net_D.state_dict(), './save_model/net_D_{}_{:4f}.pth'.format(epoch,m_psnr))
      torch.save(net_G.state_dict(), './save_model/net_G_{}_{:4f}.pth'.format(epoch,m_psnr))
      best_psnr = m_psnr










