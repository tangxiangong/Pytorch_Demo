# 创建tensor
import torch
import numpy as np

# 1. from_numpy
a = np.ones(4)
a = torch.from_numpy(a)

# 2. from list
torch.tensor([2., 3.2])
# torch.Tensor(3, 2) shape

# 未初始化的数据 empty(), Tensor(shape)
