import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import pandas as pd
from torch.optim import Adam
import logging

logger = logging.getLogger("torch_training")
hdlr = logging.FileHandler("logs/torch_traning.log", mode="w")
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)


class CreateData(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __getitem__(self, item):
        features = self.df.iloc[:, 1:6].values
        labels = self.df["label"]

        feature = torch.tensor(features)[item]
        label = torch.tensor(labels.values)[item]

        return feature, label

    def __len__(self):
        return len(self.df)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.seq = nn.Sequential(nn.Linear(5, 10), nn.Linear(10, 20), nn.Linear(20, 3))

    def forward(self, x):
        return F.softmax(self.seq(x))


class Train:
    def __init__(self):
        pass

    def train(self):
        pass

    def validate(self):
        pass

    def loss(self):
        pass

    def optimizer(self):
        pass

    def nn(self):
        pass


if __name__ == "__main__":
    cd = CreateData("features(multiclass_classify).csv")
    for i in range(10):
        print(cd.__getitem__(i))
    print(cd.__len__())
    net = Net()
    print(net)
    # res = net.forward(cd.__getitem__(5))
    # print(res)
