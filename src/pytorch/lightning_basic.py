import torch
import torch.nn.functional as F
import torchmetrics
import lightning as L
import matplotlib.pyplot as plt
import warnings

import torchvision
import wandb
import time
import os

from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.utilities.model_summary import ModelSummary


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
        self.cosine_similarity = torchmetrics.CosineSimilarity(reduction="mean")

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = self._prepare_batch(batch)
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @staticmethod
    def _prepare_batch(batch):
        x, _ = batch
        return x.view(x.size(0), -1)

    def _common_step(self, batch, batch_idx, stage: str):
        x = self._prepare_batch(batch)
        acc = self.cosine_similarity(x, self(x))
        loss = F.mse_loss(x, self(x))
        self.log_dict(
            {f"{stage}_loss": loss, f"{stage}_acc": acc}, on_step=True
        )
        return loss


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
    max_epochs=10,
    devices="auto",
    profiler=None,  # "advanced"
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

dataset = next(iter(test_loader))
autoencoder = LAutoEncoder.load_from_checkpoint(model_path)
predictions = trainer.predict(model=autoencoder, dataloaders=test_loader)
# print(ModelSummary(autoencoder, max_depth=-1))

rows, cols = 2, 5
fig = plt.figure(figsize=(12, 8))
for i in range(1, cols + 1):
    sample_id = torch.randint(low=0, high=len(test_loader), size=(1,)).item()
    img_in = test_dataset[sample_id][0]
    img_out = autoencoder(img_in.view(img_in.size(0), -1)).reshape(28, 28, 1)
    img_in = img_in.permute(1, 2, 0).numpy()
    img_out = img_out.detach().numpy()
    fig.add_subplot(1, cols, i)
    plt.axis("off")
    plt.imshow(img_in, cmap="gray")
    fig.add_subplot(2, cols, i)
    plt.axis("off")
    plt.imshow(img_out, cmap="gray")
plt.savefig("../images/mnist_sample", bbox_inches="tight")
plt.close("all")
