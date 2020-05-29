import os
import csv
import pandas as pd
from tqdm import tqdm

path = "D:\Custom-Train-Test(color)"

train_path = os.path.join(path, "Train")
test_path = os.path.join(path, "Test")

labels = [folder.split("___")[1] for folder in os.listdir(test_path)]
print(labels)
folders = [folder_name for folder_name in os.listdir(train_path)]

sample_list = []
for i, folder in tqdm(enumerate(folders, 0)):
    path = os.path.join(train_path, folder)
    for image in tqdm(os.listdir(path)):
        sample_list.append([os.path.join(path, image), labels[i]])

labels = [sample_list[i][1] for i in range(len(sample_list))]


df = pd.DataFrame(sample_list, columns=["Image path", "Label"])
df.to_csv("Leaf_disease_path.csv")
