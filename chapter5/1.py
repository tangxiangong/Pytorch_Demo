import torch

a = torch.rand(4, 4)
b = torch.rand(4, 4)

cond = torch.rand(4, 4)

torch.where(cond > 0.5, a, b)

print(torch.matmul(a, b).shape)
