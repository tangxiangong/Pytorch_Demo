# 随机初始化
import torch

a = torch.rand(3, 3)
b = torch.rand_like(a)

c = torch.randint(1, 10, [2, 3])

d = torch.randn(3, 3)  # 标准正态分布
e = torch.normal(mean=0.3, std=3, size=[2, 3])


# f = torch.full([2, 3], 3)
# g = torch.full([], 2)  # 标量
torch.arange(0, 10, 1)
torch.linspace(0,10,steps=10)
torch.ones()
torch.zeros()
torch.eye()


torch.randperm(10)  # 随机打散
