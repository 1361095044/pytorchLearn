import pandas as pd
from torchtext.datasets import AG_NEWS

train_iter = AG_NEWS(split='test')
print(train_iter)

# n = 1
# for label, line in train_iter:
#     print(label)
#     print(line)
#     n += 1
#     if n == 4:
#         break
# print('----------------')
# train_iter = AG_NEWS(split='test')
# n = 1
# for label, line in train_iter:
#     print(label)
#     print(line)
#     n += 1
#     if n == 4:
#         break
# class DataPipe(ShardingFilterIterDataPipe):
#
#     def __init__(self, source_datapipe: IterDataPipe):
#         super().__init__(source_datapipe)
#

# print(type(train_iter))
# list1 = list(train_iter)
# print(list1)

test1 = pd.DataFrame(train_iter)
print(test1)
