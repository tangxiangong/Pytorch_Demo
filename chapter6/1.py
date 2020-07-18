import torch
import numpy as np
import matplotlib.pyplot as plt


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


x = torch.tensor(-1., requires_grad=True)

optimizer = torch.optim.Adam([x], lr=1e-3)

for step in range(20000):
    pred = (x**2 - 2)**2
    optimizer.zero_grad()
    pred.backward()
    optimizer.step()

    if step % 2000 == 0:
        print('step {}: x={}, f(x)={}'.format(step, x.tolist(), pred.item()))

# x = np.arange(-6, 6, .1)
# y = np.arange(-6, 6, .1)
#
# X, Y = np.meshgrid(x, y)
#
# Z = himmelblau([X, Y])
#
# fig = plt.figure("himmelblau")
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, Z)
# ax.view_init(60, -30)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# plt.show()
