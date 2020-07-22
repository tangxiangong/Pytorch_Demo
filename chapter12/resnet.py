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
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(self.extra(x) + out)

        return out

class ResNet18(nn.Module):
    def __init__(self, num_class):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(16)
        )
        self.block1 = ResBlock(16, 32, stride=3)
        self.block2 = ResBlock(32, 64, stride=3)
        self.block3 = ResBlock(64, 128, stride=2)
        self.block4 = ResBlock(126, 256, stride=2)
        self.output = nn.Linear(256*2*2, num_class)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = self.block1(out1)
        out3 = self.block2(out2)
        out4 = self.block3(out3)
        out5 = self.block4(out4)
        out6 = x.view(out5.size(0), -1)
        out = self.output(out6)

        return out



