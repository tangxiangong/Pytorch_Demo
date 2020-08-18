#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/8/17
# @Author  : Tang Xiangong
# @Contact : tangxg16@lzu.edu.cn
# @File    : layer.py
import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.Relu()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

    def init_gates(self):
        self.gates = nn.Parameter(torch.ones(self.conv.out_channels))

    def get_gates(self):
        return [self.gates]

    def gated_forward(self, *input, **kwargs):
        out = self.conv(input[1])
        out = self.bn(out)
        out = self.gates.view(1, -1, 1, 1) * out
        out = self.relu(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel,
                 kernel_size=3, stride=1, padding=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size, stride, padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_channel, out_channel, kernel_size, 1, padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()
        self.shortcut = nn.Sequential()
        if stride != 1 or out_channel != in_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride, 0, bias=bias),
                nn.BatchNorm2d(out_channel))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu2(out)
        return out

    def init_gates(self):
        self.gates = nn.Parameter(torch.ones(self.conv1.out_channels))

    def get_gates(self):
        return [self.gates]

    def gated_forward(self, *input, **kwargs):
        out = self.conv1(input[1])
        out = self.bn1(out)
        out = self.gates.view(1, -1, 1, 1) * out
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(input[1])
        out = self.relu2(out)
        return out


def main():
    pass


if __name__ == '__main__':
    main()
