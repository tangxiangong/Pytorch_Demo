import torch
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return (x ** 2 - 2) ** 2


x = torch.tensor(1., requires_grad=True)

optimizer = torch.optim.Adam([x], lr=0.001)

loss = np.zeros(20000)
root = np.zeros(20000)

for step in range(20000):
    y = f(x)
    root[step] = x.item()
    # loss[step] = y.item()
    optimizer.zero_grad()
    y.backward()
    optimizer.step()

plt.figure()
plt.plot(root - np.sqrt(2))
plt.show()
