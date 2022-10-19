import torch
import torchvision.models

vgg16 = torchvision.models.vgg16(pretrained=False)
# # 保存方式一  模型结构+参数
# torch.save(vgg16, 'vgg16_method.pth')
# 保存方式二    以字典形式保存网络的参数  推荐
torch.save(vgg16.state_dict(), 'vgg16_method2.pth')