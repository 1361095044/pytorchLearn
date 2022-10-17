import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10('./val/data', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = Linear(196608, 10)

    # 重写方法forward
    def forward(self, input):
        output = self.linear1(input)
        return output


n = NeuralNetwork()

for data in dataloader:
    imgs, targets = data
    output = torch.flatten(imgs)  # 变成一行
    output = n(output)
    # print(output.shape)
