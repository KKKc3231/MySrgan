# -*- coding: utf-8 -*-
# @Time : 2022/10/23 16:11
# @Author : KKKc
# @FileName: save_model.py
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, discriminator=False, use_act=True, use_bn=True, **kwargs):
        super(ConvBlock, self).__init__()
        self.use_act = use_act
        self.use_bn = use_bn
        self.cnn = nn.Conv2d(in_channel, out_channel, **kwargs, bias=not use_bn)  # 用bn就不加bias，因为加了bn，bias就不起作用了
        self.bn = nn.BatchNorm2d(out_channel) if use_bn else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=False) if discriminator else nn.PReLU(
            num_parameters=out_channel)  # PReLU为带参数的relu

    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))
        # return self.act(self.cnn(x)) if self.use_act else self.cnn(x) # 不用bn


# 一个基本的残差模块：一个带激活函数，一个不带激活函数，Generator后续会使用到ResBlock
class ResBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResBlock, self).__init__()
        self.block1 = ConvBlock(in_channel, in_channel, kernel_size=3, stride=1, padding=1)  # 带激活函数
        self.block2 = ConvBlock(in_channel, in_channel, use_act=False, kernel_size=3, stride=1, padding=1)  # 不带激活函数
        # self.block3 = ConvBlock(in_channel, in_channel, use_act=False, kernel_size=3, stride=1, padding=1)  # 不带激活函数
        # self.block4 = ConvBlock(in_channel, in_channel, use_act=False, kernel_size=3, stride=1, padding=1)  # 不带激活函数

    def forward(self, x):
        # 四个bloack
        out = self.block1(x)
        output = self.block2(out)
        # output1 = self.block3(output)  #
        # output2 = self.block4(output1)
        return output + x  # 残差连接


# 上采样
class Upsample(nn.Module):
    def __init__(self, in_channel, up_scales):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(  # out_channels的维度为in_c*up**2，为后续的pixelshuffle做铺垫
            in_channels=in_channel, out_channels=in_channel * up_scales ** 2, kernel_size=3, stride=1, padding=1)
        self.ps = nn.PixelShuffle(up_scales)
        self.act = nn.PReLU(num_parameters=in_channel)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


# 生成器
class Generator(nn.Module):
    def __init__(self, in_channel=3, num_block=8):  # in_channel=3:输入彩色图像，num_block：ResBlock的堆叠数量
        super(Generator, self).__init__()
        # 残差前的没有归一化，padding = kernel_size // 2
        self.head = ConvBlock(in_channel, out_channel=64, use_bn=False, kernel_size=9, stride=1, padding=4)
        self.res = nn.Sequential(*[ResBlock(in_channel=64) for _ in range(num_block)])  # *为取到数组中的所有的值
        self.mid1 = nn.Sequential(
            ResBlock(in_channel=64),
            ResBlock(in_channel=64),
            ResBlock(in_channel=64),
            ResBlock(in_channel=64),
            ResBlock(in_channel=64)
        )
        # 残差后的卷积加BN
        self.mid2 = ConvBlock(in_channel=64, out_channel=64, use_act=False, kernel_size=3, stride=1, padding=1)
        # PixelShuffle上采样
        self.tail = nn.Sequential(
            Upsample(in_channel=64, up_scales=2),
            Upsample(in_channel=64, up_scales=2)
            # Upsample(in_channel=64, up_scales=2)
            # Upsample(in_channel=64, up_scales=2)
        )
        self.end = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        out0 = self.head(x)
        out1 = self.mid1(out0)
        out2 = out0 + self.mid2(out1)
        out3 = self.tail(out2)
        out = self.end(out3)
        return torch.tanh(out)


# 判别器
class Discriminator(nn.Module):
    def __init__(self, in_ch=3, fetures=[64, 64, 128, 128, 256, 256, 512, 512]):
        super(Discriminator, self).__init__()
        blocks = []
        for idx, feture in enumerate(fetures):
            blocks.append(
                ConvBlock(
                    in_ch,
                    out_channel=feture,
                    discriminator=True,
                    use_act=True,
                    use_bn=False if idx == 0 else True, # 第一层的卷积后面没有BN层
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1
                )
            )
            in_ch = feture
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((12,12)), # 使特征图的大小为12x12，图像大小如果为196x196，则刚好，如果大一点的话，确保能够训练
            nn.Flatten(), # 拉成一个维度
            nn.Linear(12*12*512,1024),
            nn.LeakyReLU(0.2,inplace=False),
            nn.Linear(1024,1)
        )

    def forward(self,x):
        x = self.blocks(x)
        out = self.classifier(x)
        return out


# x = torch.randn(5, 3, 24, 24)
# G_net = Generator()
# D_net = Discriminator()
#
# gen_out = G_net(x)
# dis_out = D_net(gen_out)
# print(gen_out.shape)
# print(dis_out.shape)
# print(dis_out)
