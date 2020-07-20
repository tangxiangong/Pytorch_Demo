# 自定义层
import torch
import torch.nn as nn


class MyLinear(nn.Module):
    def __init__(self, inp, outp):
        super(MyLinear, self).__init__()

        self.w = nn.Parameter(torch.randn(outp, inp))
        self.b = nn.Parameter(torch.randn(outp))

    def forward(self, x):
        return x @ self.w.t() + self.b
