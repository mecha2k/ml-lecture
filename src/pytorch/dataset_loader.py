import torch
from torch.utils.data import Dataset, DataLoader


class myDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return {"data": self.data[index], "label": self.label[index]}

    def __len__(self):
        return len(self.data)


data = ["Happy", "Amazing", "Sad", "Unhappy", "Glum"]
labels = ["positive", "positive", "negative", "negative", "neutral"]

MyDataset = myDataset(data, labels)
myDataloader = DataLoader(MyDataset, batch_size=2, shuffle=True)

for dataset in myDataloader:
    print(dataset)
