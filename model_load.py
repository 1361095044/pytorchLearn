import torch


# # 按照方式一保存进行加载
# model = torch.load('./vgg16_method.pth')
# print(model)
# 按照方式二保存进行加载  推荐
import torchvision.models
model = torch.load('./vgg16_method2.pth')
print(model)
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(model)
print(vgg16)
