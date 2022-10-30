# SRGAN学习记录📚

srgan是将生成对抗网络gan引入超分，使用对抗的思想来做超分的任务。

## 网络结构

- `Discriminator`

![](https://github.com/KKKc3231/MySrgan/blob/main/Fic/Discriminator.png)

`HR:`原始的高清图像

`SR:`经过网络超分后的图像

SRGAN的D输入为HR或SR，然后进行分类即可

- `Generator`

![](https://github.com/KKKc3231/MySrgan/blob/main/Fic/Generator.png)

G的几个模块：

`Residual blocks:`残差模块，加大网络的深度

`Upsampe（Pixelshuffle）:`上采样，补充亚像素，以扩大两倍为例，N x C*C x H x W  -->  N  x  C x 2H x 2W

需要注意的是G的卷积中`padding=kernel_size // 2`，目的是确保卷积过后图像的大小不变，只增加通道数

## loss

- `Discriminator`

D的loss分为对真实图片的bce_loss和对生成图片的bce_loss，外加梯度惩罚gp

- `Generator`

G的loss分为真实图像和生成图像的mse_loss，生成器的对抗损失gan_loss，和VGG特征损失

GAN的loss可以参考这位博主的博客：

[GAN的Loss的比较研究——传统GAN的Loss的理解_ChaoFeiLi的博客-CSDN博客_gan loss](https://blog.csdn.net/ChaoFeiLi/article/details/110431040?ops_request_misc=&request_id=&biz_id=102&utm_term=gan损失和交叉熵&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-5-110431040.142^v59^pc_rank_34_2,201^v3^add_ask&spm=1018.2226.3001.4187)

## 数据集

数据集用的B200

超分数据集可以参考下面这位博主的博客：

[最全超分辨率（SR）数据集介绍以及多方法下载链接_呆呆象呆呆的博客-CSDN博客_manga109数据集](https://blog.csdn.net/qq_41554005/article/details/116466156)

- `tranforms`

```python
# 图像处理操作，包括随机裁剪，转换张量，且不需要两次ToTensor()
H_transform = transforms.Compose(
    [
        transforms.RandomCrop(96), # 超分后的图像大小 96 x 96
        transforms.ToTensor()
    ]
)

L_transform = transforms.Compose(
    [
        transforms.Resize((24,24)), # 改成默认为BICUBIC了（我在源码里面改了一下默认）
        transforms.Normalize(mean=[0,0,0],std=[1,1,1]),
    ]
)
```

## train

`python train.py`即可

## result

超分结果在`result`文件夹中

- SR-IR

![](https://github.com/KKKc3231/MySrgan/blob/main/result/SR-IR.png)

- HR-SR

![](https://github.com/KKKc3231/MySrgan/blob/main/result/HR-SR.png)


## more

持续训练中~

- 加大了residual block的数量
- 去掉了BN层
- 加大超分后的图像大小（可能会爆内存）





