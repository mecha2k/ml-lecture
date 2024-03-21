import torch
import torch.nn.functional as F
import lightning as L
import matplotlib.pyplot as plt
import warnings
import os

from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)


class LAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


dataset = MNIST("../data", download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

fig = plt.figure(figsize=(12, 8))
rows, cols = 3, 3
for i in range(1, rows * cols + 1):
    sample_id = torch.randint(len(train_loader.dataset), size=(1,)).item()
    img, label = train_loader.dataset[sample_id]
    fig.add_subplot(rows, cols, i)
    plt.axis("off")
    plt.imshow(img.squeeze().numpy(), cmap="gray")
plt.savefig("mnist_sample", bbox_inches="tight")

trainer = L.Trainer(max_epochs=1, max_steps=32)
autoencoder = LAutoEncoder(Encoder(), Decoder())
trainer.fit(model=autoencoder, train_dataloaders=train_loader)


# Under the hood, the Lightning Trainer runs the following training loop on your behalf
# autoencoder = LAutoEncoder(Encoder(), Decoder())
# optimizer = autoencoder.configure_optimizers()
#
# for batch_idx, batch in enumerate(train_loader):
#     loss = autoencoder.training_step(batch, batch_idx)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
