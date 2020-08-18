import torch
import visdom
import torchvision
from torch import nn
from torch import optim
from resnet import ResNet18
from pokemon import Pokemon
from torch.utils.data import DataLoader

batch = 32
lr = 1e-3
epochs = 10

device = torch.device("cuda")
torch.manual_seed(1234)

train_db = Pokemon('pokemon', 224, mode='train')
val_db = Pokemon('pokemon', 224, mode='val')
test_db = Pokemon('pokemon', 224, mode='test')
train_loader = DataLoader(train_db, batch_size=batch, shuffle=True, num_workers=4)
val_loader = DataLoader(val_db, batch_size=batch, num_workers=2)
test_loader = DataLoader(test_db, batch_size=batch, num_workers=2)


def evalute(model, loader):
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total


def main():
    model = ResNet18(5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()

    best_acc, best_epoch=0,0
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), 'best.mdl')

    print("best_acc: ", best_acc, "best_epoch: ", best_epoch)
    model.load_state_dict(torch.load('best.mdl'))

    test_acc = evalute(model, test_loader)
    print("test", test_acc)

if __name__ == '__main__':
    main()
