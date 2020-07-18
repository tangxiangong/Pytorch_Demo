# 全连接层

import torch
import torch.nn as nn
# import torch.nn.functional as F


# layer1 = nn.Linear(784, 200)
# layer2 = nn.Linear(200, 200)
# layer3 = nn.Linear(200, 10)

class MxNet(nn.Module):
    def __init__(self):
        super(MxNet, self).__init__()
        self.model = nn.Sequential(nn.Linear(784, 200), nn.ReLU(inplace=True), nn.Linear(200, 200),
                                   nn.ReLU(inplace=True), nn.Linear(200, 10), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.model(x)

