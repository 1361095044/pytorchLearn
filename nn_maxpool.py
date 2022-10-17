# 最大池化操作的步长是kernel_size

import torch
import torch.nn.functional as F
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 1111111111
# input0 = torch.tensor([[1, 2, 0, 3, 1],
#                        [0, 1, 2, 3, 1],
#                        [1, 2, 1, 0, 0],
#                        [5, 2, 3, 1, 1],
#                        [2, 1, 0, 1, 1]], dtype=torch.float32)
#
# # -1自动计算channel1的数量
# input0 = torch.reshape(input0, (-1, 1, 5, 5))
dataset = torchvision.datasets.CIFAR10('./val/data', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    # 重写方法forward
    def forward(self, input):
        output = self.maxpool1(input)
        return output


neuralNetwork = NeuralNetwork()
writer = SummaryWriter('./logs_maxpool')
step = 0
# 11111111
# output = neuralNetwork(input0)
# print(output)
for data in dataloader:
    imgs, targets = data
    writer.add_images('input', imgs, step)
    # 最大池化不会改变channel
    output = neuralNetwork(imgs)
    writer.add_images('output', output, step)
    step = step + 1
    print(step)

writer.close()

