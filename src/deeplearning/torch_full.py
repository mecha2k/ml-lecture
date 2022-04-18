import torch

print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"{device} is available in torch")


def torch_variables():
    x = torch.zeros(size=(4, 2), dtype=torch.int32)
    x = torch.tensor(data=[3.0, 2.4])
    print(x)


if __name__ == "__main__":
    torch_variables()
