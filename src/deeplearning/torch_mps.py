import torch
import sys
import platform


print(torch.__version__)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else device)
print(f"{device} is available in torch")

print(sys.version)
print(platform.platform())

sample = torch.randn(2, 2).to(device)
print(sample)