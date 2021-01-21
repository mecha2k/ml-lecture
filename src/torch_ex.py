import torch
import torch.nn as nn
import torch.nn.functional as F

from icecream import ic

x = torch.ones(2, 2, requires_grad=True)
ic(x)

y = x + 2
ic(y)
ic(y.grad_fn)

z = y * y * 3
out = z.mean()
ic(z, out)

out.backward()
ic(x.grad)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 입력 이미지 채널 1개, 출력 채널 6개, 3x3의 정사각 컨볼루션 행렬
        # 컨볼루션 커널 정의
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 아핀(affine) 연산: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6은 이미지 차원에 해당
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # (2, 2) 크기 윈도우에 대해 맥스 풀링(max pooling)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 크기가 제곱수라면 하나의 숫자만을 특정
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 배치 차원을 제외한 모든 차원
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
ic(net)

params = list(net.parameters())
ic(len(params))
ic(params[0].size())  # conv1의 .weight

input = torch.randn(1, 1, 32, 32)
out = net(input)
ic(out)

net.zero_grad()
out.backward(torch.randn(1, 10))

output = net(input)
target = torch.randn(10)  # 예시를 위한 임의의 정답
target = target.view(1, -1)  # 출력과 같은 shape로 만듦
criterion = nn.MSELoss()

loss = criterion(output, target)
ic(loss)

ic(loss.grad_fn)  # MSELoss
ic(loss.grad_fn.next_functions[0][0])  # Linear
ic(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

net.zero_grad()     # 모든 매개변수의 변화도 버퍼를 0으로 만듦

ic('conv1.bias.grad before backward')
ic(net.conv1.bias.grad)

loss.backward()

ic('conv1.bias.grad after backward')
ic(net.conv1.bias.grad)

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

import torch.optim as optim

# Optimizer를 생성합니다.
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 학습 과정(training loop)에서는 다음과 같습니다:
optimizer.zero_grad()   # 변화도 버퍼를 0으로
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # 업데이트 진행

