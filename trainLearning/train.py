import torch.optim
import torchvision.datasets
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter

from model import *
from torch.utils.data import DataLoader

# 准备数据集
train_data = torchvision.datasets.CIFAR10('./val/data', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10('./val/data', train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())
train_data_size = len(train_data)
test_data_size = len(test_data)
print(train_data_size, test_data_size)
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
tudui = Tudui()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.01  # 1e-2   = 0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 记录训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter('./logs_train')

for i in range(epoch):
    tudui.train()
    print(f'第{i + 1}轮')
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)
        # 梯度清零以及优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f'训练次数{total_train_step},loss:{loss.item()}')
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # 测试不需要对参数进行调整
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print(f'整体测试集的loss：{total_test_loss}')
    print(f'正确率：{total_accuracy/test_data_size}')
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', total_accuracy/test_data_size, total_test_step)
    total_test_step += 1
    # 保存模型
    torch.save(tudui, f'tudui_{i}.pth')

writer.close()
