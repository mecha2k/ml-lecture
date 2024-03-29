import torch
import torch.nn.functional as F
import lightning
import warnings
import os

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from time import time

warnings.filterwarnings("ignore", category=UserWarning)
print(lightning.__version__)


class LitMNIST(LightningModule):
    def __init__(self, data_dir="../data/mnist", hidden_size=64, learning_rate=2e-4):
        super().__init__()
        self.data_dir = data_dir
        self.num_classes = 10
        self.dims = (1, 28, 28)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        channels, width, height = self.dims
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=128)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=128)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=128)


model = LitMNIST()
trainer = Trainer(
    max_epochs=0,
    accelerator="auto",
    devices="auto",
    callbacks=[TQDMProgressBar(refresh_rate=20)],
)

current = time()
trainer.fit(model)
print("elapsed time : ", time() - current)
