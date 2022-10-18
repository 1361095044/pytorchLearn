import torch.optim
import torchvision
from torch import nn
from torch.nn import Sequential, Linear, Conv2d, MaxPool2d, Flatten
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10('./val/data', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )

    def forward(self, x):
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()
tudui = Tudui()
# 优化器
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)

for i in range(20):
    running_loss = 0
    for data in dataloader:
        imgs, target = data
        output = tudui(imgs)
        # print(output)
        # print(target)
        result_loss = loss(output, target)
        # 梯度清零
        optim.zero_grad()
        # 反向传播
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)
