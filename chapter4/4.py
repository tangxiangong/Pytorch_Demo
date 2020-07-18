# 切片
import torch

a = torch.rand(4, 3, 28, 28)

print(a[0].shape)
print(a[0,0,2,4])
torch.masked_select()
torch.take()
