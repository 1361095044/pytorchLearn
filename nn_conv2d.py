import torch
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch import nn

dataset = torchvision.datasets.CIFAR10('../pytorch_learn/val/data', train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=False)

dataloader = DataLoader(dataset, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3)

    # 重写方法forward
    def forward(self, x):
        x = self.conv1(x)
        return x


neuralNetwork = NeuralNetwork()
# print(neuralNetwork)
n = 0
for data in dataloader:
    imgs, targets = data
    # 等价于 neuralNetwork.forward(imgs)
    output = neuralNetwork(imgs)
    print(imgs.shape)
    print(output.shape)
    n += 1
    while n == 2:
        break


