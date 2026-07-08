# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path
from base import *

import torchvision.utils as vutils

# 하이퍼파라미터 설정 (기존 dcgan_faces.py 기준)
dataroot = Path("../data/celeba")
epochs = 5
batch_size = 256
lr = 0.0002
beta1 = 0.5
workers = 4

image_size = 64
nc = 3  # 컬러 이미지 채널 수
nz = 100  # 잠재 벡터(z) 크기
ngf = 64  # 생성기 피처 맵 크기
ndf = 64  # 판별기 피처 맵 크기
ngpu = 1  # 사용 가능한 GPU 수 (0인 경우 CPU 모드)


# 가중치 초기화 함수
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# 생성기(Generator) 정의
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 입력은 Z 잠재 벡터, 합성곱 레이어로 전달
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 상태 크기: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 상태 크기: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 상태 크기: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 상태 크기: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # 상태 크기: (nc) x 64 x 64
        )

    def forward(self, x):
        return self.main(x)


# 판별기(Discriminator) 정의
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 입력은 (nc) x 64 x 64 크기의 이미지
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태 크기: (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태 크기: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태 크기: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태 크기: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x)


def main():
    # 데이터셋 및 데이터로더 생성
    # 데이터셋 경로가 존재하고 올바른 폴더 구조를 가졌을 때만 로드하도록 예외 처리
    dataset_available = False

    if dataroot.exists():
        try:
            dataset = datasets.ImageFolder(
                root=dataroot,
                transform=transforms.Compose(
                    [
                        transforms.Resize(image_size),
                        transforms.CenterCrop(image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=workers,
            )
            dataset_available = True
        except FileNotFoundError as e:
            print(f"[경고] 데이터셋 폴더 구조가 올바르지 않습니다: {e}")
            print(
                "PyTorch ImageFolder를 사용하려면 root 디렉토리 하위에 최소 하나의 클래스 서브디렉토리가 존재해야 합니다."
            )
            print(
                f"예시 경로 구조: {dataroot.resolve()}/img_align_celeba/*.jpg"
            )
            dataloader = []
    else:
        print(f"[경고] 데이터셋 경로를 찾을 수 없습니다: {dataroot.resolve()}")
        print(
            "학습을 진행하려면 Celeb-A 데이터셋을 해당 경로에 압축 해제해야 합니다."
        )
        dataloader = []

    # 모델 인스턴스 생성 및 가중치 초기화
    # device는 base.py에서 정의된 값을 사용합니다.
    netG = Generator(ngpu).to(device)
    if (device.type == "cuda") and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    netG.apply(weights_init)

    netD = Discriminator(ngpu).to(device)
    if (device.type == "cuda") and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    netD.apply(weights_init)

    # 손실 함수 및 옵티마이저 설정
    criterion = nn.BCELoss()

    # 생성기의 학습 과정을 시각화하기 위한 고정된 노이즈 벡터 생성
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # 진짜 이미지와 가짜 이미지의 라벨 설정
    real_label = 1.0
    fake_label = 0.0

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # 학습 루프 실행
    if dataset_available:
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print("학습 루프를 시작합니다...")
        for epoch in range(epochs):
            for i, data in enumerate(dataloader, 0):
                ############################
                # (1) 판별기(D) 업데이트: log(D(x)) + log(1 - D(G(z))) 최대화
                ###########################
                # 진짜 이미지 배치로 학습
                netD.zero_grad()
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full(
                    (b_size,), real_label, dtype=torch.float, device=device
                )

                output = netD(real_cpu).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                # 가짜 이미지 배치로 학습
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake = netG(noise)
                label.fill_(fake_label)

                output = netD(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()

                errD = errD_real + errD_fake
                optimizerD.step()

                ############################
                # (2) 생성기(G) 업데이트: log(D(G(z))) 최대화
                ###########################
                netG.zero_grad()
                label.fill_(
                    real_label
                )  # 생성기 비용 함수를 위해 가짜 라벨을 진짜로 설정

                output = netD(fake).view(-1)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()

                optimizerG.step()

                # 학습 상태 출력 (50번 반복마다)
                if i % 50 == 0:
                    print(
                        f"[{epoch}/{epochs}][{i}/{len(dataloader)}]\t"
                        f"Loss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\t"
                        f"D(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}"
                    )

                # 추후 시각화를 위해 손실값 저장
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # fixed_noise를 사용하여 생성기의 성능을 주기적으로 기록
                if (iters % 500 == 0) or (
                    (epoch == epochs - 1) and (i == len(dataloader) - 1)
                ):
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().cpu()
                    img_list.append(
                        vutils.make_grid(fake, padding=2, normalize=True)
                    )

                iters += 1

        # 결과 디렉토리 생성 (outputs 폴더 생성)
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        # 1. 손실 추이 그래프 저장
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(output_dir / "dcgan_loss_plot.png")
        plt.close()
        print(
            f"손실 추이 그래프가 저장되었습니다: {output_dir / 'dcgan_loss_plot.png'}"
        )

        # 2. 마지막 에폭에서 생성된 가짜 이미지 시각화 결과 저장
        if len(img_list) > 0:
            plt.figure(figsize=(8, 8))
            plt.axis("off")
            plt.title("Generated Fake Images")
            plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
            plt.savefig(output_dir / "dcgan_fake_images.png")
            plt.close()
            print(
                f"생성된 가짜 이미지 결과가 저장되었습니다: {output_dir / 'dcgan_fake_images.png'}"
            )

        # 3. 진짜 이미지와 가짜 이미지 비교 저장
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(15, 15))

        # 진짜 이미지
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(
            np.transpose(
                vutils.make_grid(
                    real_batch[0].to(device)[:64], padding=5, normalize=True
                ).cpu(),
                (1, 2, 0),
            )
        )

        # 가짜 이미지
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        if len(img_list) > 0:
            plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))

        plt.savefig(output_dir / "dcgan_real_vs_fake.png")
        plt.close()
        print(
            f"진짜 이미지 대 가짜 이미지 비교가 저장되었습니다: {output_dir / 'dcgan_real_vs_fake.png'}"
        )

    else:
        print("데이터셋이 유효하지 않아 학습 및 결과 저장을 진행하지 않습니다.")


if __name__ == "__main__":
    main()
