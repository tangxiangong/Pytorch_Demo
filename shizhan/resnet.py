#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/8/17
# @Author  : Tang Xiangong
# @Contact : tangxg16@lzu.edu.cn
# @File    : resnet.py
from layer import ConvBNReLU
from layer import BasicBlock
import torch.nn as nn

cfgs = {
    'resnet20': [16] * 3 + [32] * 3 + [64] * 3,
    'resnet32': [16] * 5 + [32] * 5 + [64] * 5,
    'resnet44': [16] * 7 + [32] * 7 + [64] * 7,
    'resnet56': [16] * 9 + [32] * 9 + [64] * 9,
}

stride2_loc = {
    '20': [3, 6],
    '32': [5, 10],
    '44': [7, 14],
    '56': [9, 18]
}


class ResNet(nn.Module):
    def __init__(self, depth, cfg, num_classes=10):
        super(ResNet, self).__init__()
        self.depth = depth
        self.cfg = cfg
        self.conv = ConvBNReLU(3, 16)
        self.in_channel = 16
        self.features = self._make_layers()
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(self.in_channel, num_classes)

    def _make_layers(self):
        layers = []
        stride2_idx = stride2_loc[str(self.depth)]
        for i, n in enumerate(self.cfg):
            stride = 1
            out_channel = self.in_channel
            if i in stride2_idx:
                stride = 2
                out_channel = 2 * self.in_channel
            layers.append(BasicBlock(self.in_channel, n, out_channel, stride=stride))
            self.in_channel = out_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.features(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def resnet20(cfg=None, num_classes=10):
    if cfg is None:
        cfg = cfgs['resnet20']
    return ResNet(20, cfg, num_classes)


def resnet32(cfg=None, num_classes=10):
    if cfg is None:
        cfg = cfgs['resnet32']
    return ResNet(32, cfg, num_classes)


def resnet44(cfg=None, num_classes=10):
    if cfg is None:
        cfg = cfgs['resnet44']
    return ResNet(44, cfg, num_classes)


def resnet56(cfg=None, num_classes=10):
    if cfg is None:
        cfg = cfgs['resnet56']
    return ResNet(56, cfg, num_classes)


def main():
    pass


if __name__ == '__main__':
    main()
