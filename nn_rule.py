import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))
print(input)
dataset = torchvision.datasets.CIFAR10('./val/data', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.ReLU1 = ReLU()
        self.sigmoid1 = Sigmoid()

    # 重写方法forward
    def forward(self, input):
        # output = self.ReLU1(input)
        output = self.sigmoid1(input)
        return output


n = NeuralNetwork()
# output = n(input)
# print(output)
writer = SummaryWriter("./logs_relu")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = n(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()
