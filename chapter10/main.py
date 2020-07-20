import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):

    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_in != ch_out:
            self.extra = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride))

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out = self.extra(x) + out2
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64))
        self.block1 = ResBlock(64, 128)
        self.block2 = ResBlock(128, 256)
        self.block3 = ResBlock(256, 512)
        self.block3 = ResBlock(512, 1024)
        self.outlayer = nn.Linear(512, 10)

    def forward(self, x):
        out1 = F.relu(self.con1(x))
        out2 = self.block1(out1)
        out3 = self.block2(out2)
        out4 = self.block3(out3)
        out5 = self.block4(out4)
        out = self.outlayer(out5)
        return out


def main():
    temp = torch.randn(2, 3, 32, 32)


if __name__ == "__main__":
    main()
