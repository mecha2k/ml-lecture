import torch

print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"{device} is available in torch")

if __name__ == '__main__':
    pass