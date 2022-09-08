import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from time import time
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"{device} is available in torch")

datasets = MNIST("../data/mnist", train=True, transform=transforms.ToTensor())
mnist_train, mnist_valid = random_split(datasets, [55000, 5000])
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)


class MNISTModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)


model = MNISTModule().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)  # 내부적으로 소프트맥스 함수를 포함하고 있음.
optimizer = Adam(model.parameters(), lr=1e-4)

epochs = 1
model.train()
current = time()
for epoch in range(epochs):
    for batch in tqdm(train_loader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("elapsed time : ", time() - current)
