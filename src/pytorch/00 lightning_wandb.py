import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torchmetrics
import wandb

from torchvision.datasets import MNIST
from torchvision import transforms
from argparse import Namespace


hparams = Namespace(
    n_layer1=128,
    n_layer2=256,
    num_classes=10,
    learning_rate=0.001,
)


class LightningDataMNIST(LightningDataModule):
    def __init__(self, data_dir="../data/mnist", batch_size=256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()

        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None

    # called only once and on 1 GPU
    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    # called one ecah GPU separately - stage defines if we are at fit or test step
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_train = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=0)


class LightningMNIST(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.layer1 = nn.Linear(28 * 28, hparams.n_layer1)
        self.layer2 = nn.Linear(hparams.n_layer1, hparams.n_layer2)
        self.layer3 = nn.Linear(256, hparams.num_classes)

        self.learning_rate = hparams.learning_rate
        self.accuracy = torchmetrics.Accuracy()
        self.save_hyperparameters()

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return F.log_softmax(x, dim=1)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("val_loss", loss)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("test_loss", loss)
        return {"loss": loss}

    # def train_dataloader(self):
    #     return DataLoader(
    #         MNIST(
    #             "./data",
    #             train=True,
    #             download=True,
    #             transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
    #         ),
    #         batch_size=self.hparams.batch_size,
    #     )


wandb_logger = WandbLogger(project="MNIST")

model = LightningMNIST(hparams)
mnist_loader = LightningDataMNIST()

trainer = Trainer(logger=wandb_logger, accelerator="cpu", devices=1, max_epochs=1)
trainer.fit(model=model, train_dataloaders=mnist_loader)

wandb.finish()
