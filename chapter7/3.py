import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

batch_size = 200
learning_rate = 0.01
epochs = 10

train_loader = torch.utils.data.DataLoader(datasets.MNIST("../data", train=True, download=True,
                                                          transform=transforms.Compose([transforms.ToTensor(),
                                                                                        transforms.Normalize((0.1307,),
                                                                                                             (
                                                                                                                 0.3081,))])),
                                           batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(datasets.MNIST("../data", train=False, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=batch_size, shuffle=True)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(nn.Linear(784, 200), nn.LeakyReLU(inplace=True), nn.Linear(200, 200),
                                   nn.LeakyReLU(inplace=True), nn.Linear(200, 10), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.model(x)


device = torch.device('cuda:0')
net = MLP().to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss().to(device)
