import torch
import torch.nn.functional as F
import lightning as L
import matplotlib.pyplot as plt
import warnings
import wandb
import time
import os

from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import (
    ModelSummary,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler


torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", category=UserWarning)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28)
        )

    def forward(self, x):
        return self.l1(x)


class LAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("test_loss", loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


dataset = MNIST(
    "../data", download=True, train=True, transform=transforms.ToTensor()
)
train_size = int(len(dataset) * 0.8)
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, valid_size],
    generator=torch.Generator().manual_seed(42),
)
test_dataset = MNIST(
    "../data", download=True, train=False, transform=transforms.ToTensor()
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=True)

tensorboard_logger = TensorBoardLogger(save_dir="../data", name="tensorboards")
# wandb_logger = WandbLogger(project="lightning-wandb", save_dir="../data")
# wandb_logger.experiment.config["batch_size"] = 256

checkpoint_callback = ModelCheckpoint(
    save_top_k=10,
    verbose=True,
    monitor="val_loss",
    mode="min",
    dirpath="../data/checkpoints",
)

trainer = L.Trainer(
    max_epochs=5,
    devices="auto",
    profiler="advanced",
    logger=True,
    precision="bf16-mixed",
    default_root_dir="../data/checkpoints",
    enable_progress_bar=True,
    callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=10)],
)

model_path = "../data/checkpoints/autoencoder.ckpt"
if os.path.exists(model_path):
    autoencoder = LAutoEncoder.load_from_checkpoint(model_path)
    print(f"Model loaded from {model_path}")
else:
    autoencoder = LAutoEncoder(Encoder(), Decoder())
    print("no model found")

start = time.time()
trainer.fit(
    model=autoencoder,
    train_dataloaders=train_loader,
    val_dataloaders=valid_loader,
)
trainer.test(model=autoencoder, dataloaders=test_loader)
print(f"Time taken: {time.time() - start:.2f} seconds")
trainer.save_checkpoint(model_path)

# wandb.finish()

# Under the hood, the Lightning Trainer runs the following training loop on your behalf
# autoencoder = LAutoEncoder(Encoder(), Decoder())
# optimizer = autoencoder.configure_optimizers()
#
# for batch_idx, batch in enumerate(train_loader):
#     loss = autoencoder.training_step(batch, batch_idx)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()


fig = plt.figure(figsize=(12, 8))
rows, cols = 3, 3
for i in range(1, rows * cols + 1):
    sample_id = torch.randint(len(train_loader.dataset), size=(1,)).item()
    img, label = train_loader.dataset[sample_id]
    fig.add_subplot(rows, cols, i)
    plt.axis("off")
    plt.imshow(img.squeeze().numpy(), cmap="gray")
plt.savefig("../images/mnist_sample", bbox_inches="tight")
