import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

# 相减
loss = L1Loss(reduction='sum')
result = loss(inputs, targets)

print(result)
# 平方差
loss2 = MSELoss(reduction='mean')
result2 = loss2(inputs, targets)

print(result2)

# 交叉熵
loss3 = CrossEntropyLoss()
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))  # 三类
result3 = loss3(x, y)
print(result3)