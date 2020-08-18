import torch
import numpy as np
import matplotlib.pyplot as plt


def f(x_):
    return x_ ** 2 - 2


x = torch.tensor(1., requires_grad=True)

optimizer = torch.optim.Adam([x], lr=0.001)

loss = np.zeros(20000)
root = np.zeros(20000)
a = np.sqrt(2)

for step in range(20000):
    y = f(x)
    optimizer.zero_grad()
    loss[step] = a - x.item()
    root[step] = x.item()
    y.backward()
    optimizer.step()

plt.figure()
plt.plot(root)
plt.plot(loss)
plt.show()
