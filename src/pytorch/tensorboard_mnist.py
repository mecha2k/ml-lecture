import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.tensorboard import SummaryWriter
from time import time
from tqdm import tqdm
from pathlib import Path

import warnings
import sys
import os

warnings.filterwarnings("ignore", category=UserWarning)

torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else device)
print(f"{device} is available in torch")


epochs = 1
batch_size = 32
learning_rate = 1e-4

writer = SummaryWriter(log_dir=Path("../data/tensorboards/mnist"))

train_datasets = MNIST(
    "../data/mnist", train=True, transform=transforms.ToTensor(), download=True
)
test_datasets = MNIST(
    "../data/mnist", train=False, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=False)

examples = iter(test_loader)
example_data, example_targets = next(examples)
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(example_data[i][0], cmap="gray")
plt.savefig("../data/images/mnist.png")

img_grid = torchvision.utils.make_grid(example_data)
writer.add_image("mnist_images", img_grid)


class NeuralNet(nn.Module):
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


model = NeuralNet().to(device)
loss_fn = nn.CrossEntropyLoss()  # 내부적으로 소프트맥스 함수를 포함하고 있음.
optimizer = Adam(model.parameters(), lr=learning_rate)

writer.add_graph(model, example_data.reshape(-1, 28 * 28).to(device))

current = time()

model.train()
losses, accuracies = 0, 0
total_steps = len(train_loader)
for epoch in range(epochs):
    for i, (image, label) in enumerate(train_loader):
        image = image.reshape(-1, 28 * 28).to(device)
        label = label.to(device)
        y_pred = model(image)
        loss = loss_fn(y_pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}"
            )
            writer.add_scalar("loss", losses / 100, epoch * total_steps + i)
            losses = 0.0

print("elapsed time : ", time() - current)
writer.close()
