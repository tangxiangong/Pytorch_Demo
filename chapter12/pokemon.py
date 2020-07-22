import os
import csv
import glob
import torch
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Pokemon(Dataset):
    def __init__(self, root: str, resize: int, mode: str):
        super(Pokemon, self).__init__()
        self.root = root
        self.resize = resize
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(self.root))):
            if not os.path.isdir(os.path.join(self.root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
        self.images, self.labels = self.load_csv('pokemon.csv')
        if mode == 'train':
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == 'val':
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def load_csv(self, filename: str):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img, label])
        images_read, labels_read = [], []
        with open(os.path.join(self.root, filename), mode='r') as f:
            reader = csv.reader(f)
            for row in reader:
                image_read, label_read = row
                label_read = int(label_read)
                images_read.append(image_read)
                labels_read.append(label_read)

        assert len(images_read) == len(labels_read)
        return images_read, labels_read

    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean
        return x

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        assert 0 <= item < self.__len__()
        image, label = self.images[item], self.labels[item]
        transform = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((int(1.25 * self.resize), int(1.25 * self.resize))),
            transforms.RandomRotation(15),  # 旋转
            transforms.CenterCrop(self.resize),  # 中心裁剪
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)
        label = torch.tensor(label)
        return image, label


def main():
    import time
    import visdom
    vis = visdom.Visdom()
    poke = Pokemon('pokemon', 224, 'train')
    x, y = next(iter(poke))
    print("sample:", x.shape, y.shape)
    vis.image(poke.denormalize(x), win='sample_x', opts=dict(title='sample_x'))
    loader = DataLoader(poke, batch_size=32, shuffle=True)
    for x, y in loader:
        vis.images(poke.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        vis.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
        time.sleep(10)


if __name__ == '__main__':
    main()
