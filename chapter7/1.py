# 多分类 实战
import torch
from torch import *


w1, b1 = torch.randn(200, 784, requires_grad=True), torch.zeros(200, requires_grad=True)
w2, b2 = torch.randn(200, 200, requires_grad=True), torch.zeros(200, requires_grad=True)
w3, b3 = torch.randn(10, 200, requires_grad=True), torch.zeros(200, requires_grad=True)

def forward(x):
    x1 = x@w1.t() + b1
    l1 = F.relu(x1)
    x2 = l1@w2.t() + b2
    l2 = F.relu(x2)
    x3 = l2@w3.t() + b3
    l3 = F.relu(x3)
    return l3

