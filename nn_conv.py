import torch
import torch.nn.functional as F
input0 = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

# 转换一下shape
input0 = torch.reshape(input0, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))
print(input0.shape)
print(kernel.shape)
output = F.conv2d(input0, kernel, stride=1)

print(output)