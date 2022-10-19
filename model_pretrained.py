import torchvision.models
from torch.nn import Linear
# 加载现有的模型   下载到C盘去了
vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)
# 添加新的
# vgg16_true.add_module('add_linear', Linear(1000, 10))
# print(vgg16_true)
# 添加新的进内层
vgg16_true.classifier.add_module('add_linear', Linear(1000, 10))
print(vgg16_true)
# 修改module
print(vgg16_false)
vgg16_false.classifier[6] = Linear(4096, 10)
print(vgg16_false)