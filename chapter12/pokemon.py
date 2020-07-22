import os
import csv
import glob
import torch
import random
from torch.utils.data import Dataset


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

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


def main():
    bd = Pokemon('pokemon', 224, 'train')


if __name__ == '__main__':
    main()
