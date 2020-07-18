# 张量数据类型，无 string 类型，one-hot，embedding：word2vec 等
import torch
import numpy as np

a = torch.randn(2, 3)
print(a.type())
isinstance(a, torch.FloatTensor)

a = a.cuda()

b = torch.tensor(1.)
b.shape
b.size()
b.dim()

v1 = torch.tensor([1.])
v2 = torch.tensor([1, 2, 3])
torch.FloatTensor([1.1, 2.2])
torch.FloatTensor(1)  # 1-d len:1
data = np.ones(2)
torch.from_numpy(data)

a = torch.ones(2)
