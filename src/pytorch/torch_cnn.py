import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


print(torch.__version__)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda:0")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(
        f"VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB"
    )
elif device.type == "mps":
    print("Apple Silicon MPS 사용 중")
    print(f"MPS 사용 가능 여부: {torch.backends.mps.is_available()}")
    print(f"MPS 빌드 포함 여부: {torch.backends.mps.is_built()}")
print("CUDA version : ", torch.version.cuda)


batch_size = 256

train_dataset = datasets.CIFAR10(
    root="../data", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.CIFAR10(
    root="../data", train=False, transform=transforms.ToTensor(), download=True
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv1(x)


# inputs = torch.randn(64, 3, 512, 512)
# model = Net().to(device)
# output = model(inputs.to(device))
# print(output.shape)


# # Hyper-parameters
# num_epochs = 5
# batch_size = 64
# learning_rate = 0.001

# # MNIST dataset
# train_dataset = torchvision.datasets.MNIST(
#     root="./data",
#     train=True,
#     transform=transforms.ToTensor(),
#     download=True,
# )

# test_dataset = torchvision.datasets.MNIST(
#     root="./data",
#     train=False,
#     transform=transforms.ToTensor(),
# )

# # Data loader
# train_loader = torch.utils.data.DataLoader(
#     dataset=train_dataset, batch_size=batch_size, shuffle=True
# )

# test_loader = torch.utils.data.DataLoader(
#     dataset=test_dataset, batch_size=batch_size, shuffle=False
# )


# # Convolutional neural network (two convolutional layers)
# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)


# model = ConvNet().to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Train the model
# total_step = len(train_loader)
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.to(device)
#         labels = labels.to(device)

#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if (i + 1) % 100 == 0:
#             print(
#                 f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}"
#             )

# # Test the model
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print(
#         f"Test Accuracy of the model on the 10000 test images: {(100 * correct / total):.2f}%"
#     )

# # Save the model checkpoint
# torch.save(
#     model.state_dict(),
#     os.path.join(
#         sys.path[0], "model.ckpt"
#     ),  # sys.path[0] is the path of the current file
# )

# # Save the model config
# with open(os.path.join(sys.path[0], "model_config.json"), "w") as f:
#     json.dump(model.config, f)  # model.config is a dictionary

# # Save the model architecture
# with open(os.path.join(sys.path[0], "model_architecture.txt"), "w") as f:
#     f.write(str(model))  # str(model) is the model architecture
