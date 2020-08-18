#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/8/17
# @Author  : Tang Xiangong
# @Contact : tangxg16@lzu.edu.cn
# @File    : vgg.py
from layer import ConvBNReLU
import torch.nn as nn

cfgs = {
    'vgg16_bn': [
        64, 64, 128, 128,
        256, 256, 256,
        512, 512, 512,
        512, 512, 512
    ],
    'vgg19_bn': [
        64, 64, 128, 128,
        256, 256, 256, 256,
        512, 512, 512, 512,
        512, 512, 512, 512,
    ]
}

maxpool_loc = {
    '16': [1, 3, 6, 9],
    '19': [1, 3, 7, 11]
}


class VGG(nn.Module):
    def __init__(self, depth, cfg, num_classes=10):
        super(VGG, self).__init__()
        self.depth = depth
        self.cfg = cfg
        self.features = self._make_layers()
        self.avgpool = nn.AvgPool2d(2)
        self.classifier = nn.Linear(cfg[-1], num_classes)

    def _make_layers(self):
        maxpool_idx = maxpool_loc[str(self.depth)]
        layers = []
        in_channel = 3
        for i, n in enumerate(self.cfg):
            layers.append(ConvBNReLU(in_channel, n))
            if i in maxpool_idx:
                layers.append(nn.MaxPool2d(2, 2))
            in_channel = n
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def vgg16_bn(cfg=None, num_classes=10):
    if cfg is None:
        cfg = cfgs['vgg16_bn']
    return VGG(16, cfg, num_classes)


def vgg19_bn(cfg=None, num_classes=10):
    if cfg is None:
        cfg = cfgs['vgg19_bn']
    return VGG(19, cfg, num_classes)


def main():
    pass


if __name__ == '__main__':
    main()