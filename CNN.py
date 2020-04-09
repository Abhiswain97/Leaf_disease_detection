import torch
from torchvision.models import resnet50
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import numpy as np
import pandas as pd
import os
import cv2
from keras.utils import to_categorical

net = resnet50(pretrained=True)
print(net)


class Data(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        self.df = pd.read_csv(path)
        self.label_map = {}

    def __getitem__(self, item):
        images = self.df['Image path']
        labels = self.df['Label']

        for i, label in enumerate(labels.unique()):
            self.label_map[label] = i

        labels = labels.map(self.label_map)

        image = torch.from_numpy(cv2.imread(os.path.abspath(images[item])))
        label = torch.tensor(labels[item])

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

    def __len__(self):
        return len(self.df)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)


def valid(epoch, f):
    net.eval()
    for i, (features, labels) in enumerate(val_loader, 1):
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()

        outputs = net(features.float())
        loss = criterion(outputs, labels.long())

        print('Epoch: {}. Batch: {} Valid Loss: {}'.format(epoch + 1, i, loss.item()))
        f.write('\n Epoch: {}. Batch: {} Valid Loss: {}'.format(epoch + 1, i, loss.item()))


def train(epoch, f):
    net.train()
    for i, (features, labels) in enumerate(train_loader, 1):
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        outputs = net(features.float())
        loss = criterion(outputs, labels.long())

        print('Epoch: {}. Batch: {} Loss: {}'.format(epoch + 1, i, loss.item()))
        f.write('\n Epoch: {}. Batch: {} Training Loss: {}'.format(epoch + 1, i, loss.item()))

        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    path: str = 'Leaf_disease_path.csv'

    data_transforms = dict(train=transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]), val=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))

    validation_split = 0.8
    dataset_size = Data(path).__len__()
    print(Data(path).__getitem__(1300))
    # exit(0)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    train_indices, valid_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    dataset = Data(path, transform=data_transforms['train'])

    train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=32, sampler=valid_sampler)

    for i in range(10):
        f = open('Training_logs(torch).txt', 'a+')
        train(i, f)
        valid(i, f)
