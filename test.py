# -*- coding: utf-8 -*-
# @Time : 2022/10/29 17:15
# @Author : KKKc
# @FileName: test.py

from new_model import Generator,Discriminator
import torch
from PIL import Image
import torchvision.transforms as transform


model = Generator()
model_D = Discriminator()
model.load_state_dict(torch.load('save_model/new_net_G_1170.pth',map_location='cpu'))
IR_image = Image.open('./test_image/Girl_lr.png')

trans_Tensor = transform.Compose([
    transform.ToTensor()
])

trans_PIL = transform.Compose([
    transform.ToPILImage()
])

image = trans_Tensor(IR_image)
image = torch.unsqueeze(image,dim=0) # 升维度

out = model(image)
fake_dis = model_D(out)
print(fake_dis.shape)
out = torch.squeeze(out,dim=0) # 降维度
print(out.shape)
print(out)


SR_Image = trans_PIL(out)
SR_Image.show()
SR_Image.save("save.jpg")

# torch.save(out, "out.jpg")
