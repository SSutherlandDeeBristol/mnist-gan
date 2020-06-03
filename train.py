import argparse
import os
import time
from multiprocessing import cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn
import torchvision.datasets
import torchvision.utils as vutils
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from discriminator import Discriminator
from generator import Generator

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a simple CNN on CIFAR-10",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--batch-size", default=128)
parser.add_argument("--learning-rate", default=2e-4, type=float)
parser.add_argument("--epochs", default=15)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def main(args):
    transform = transforms.ToTensor()

    args.dataset_root.mkdir(parents=True, exist_ok=True)

    train_dataset = torchvision.datasets.MNIST(
        args.dataset_root, train=True, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=8,
    )

    log_dir = get_summary_writer_log_dir(args)

    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )

    noise_vector_size = 100
    real_label = 1
    fake_label = 0

    fixed_noise = torch.randn(16,100,1,1).to(DEVICE)
    fixed_noise_generator_output = dict()

    generator = Generator(noise_vector_size,28,28,1).to(DEVICE)

    discriminator = Discriminator(28,28,1).to(DEVICE)

    criterion = nn.BCELoss()

    generator_optimiser = optim.Adam(generator.parameters(), 0.0002,(0.5,0.999))
    discriminator_optimiser = optim.Adam(discriminator.parameters(), 0.0002, (0.5,0.999))

    try:
        os.mkdir("generated_digits")
    except FileExistsError:
        pass

    step = 0

    for i in range(args.epochs):
        for j, (batch,labels) in enumerate(train_loader):
            batch = batch.to(DEVICE)
            labels = labels.to(DEVICE)

            discriminator.zero_grad()
            d_real_output = discriminator.forward(batch).view(-1)
            Dx = d_real_output.mean().item()

            real_labels = torch.full(labels.shape, real_label, device=DEVICE)
            d_real_error = criterion(d_real_output, real_labels)
            d_real_error.backward()

            noise = torch.randn(len(labels), noise_vector_size, 1, 1).to(DEVICE)

            fake_data = generator.forward(noise)
            fake_labels = torch.full(labels.shape, fake_label, device=DEVICE)

            d_fake_output = discriminator.forward(fake_data.detach()).view(-1)
            d_fake_error = criterion(d_fake_output, fake_labels)
            d_fake_error.backward()
            DGz = d_fake_output.mean().item()

            d_error = d_fake_error + d_real_error

            discriminator_optimiser.step()

            generator.zero_grad()

            g_real_labels = torch.full(labels.shape, real_label, device=DEVICE)

            d_output = discriminator.forward(fake_data).view(-1)

            g_error = criterion(d_output, g_real_labels)

            g_error.backward()

            generator_optimiser.step()

            print(f"epoch: {i}, step: {j+1}/{len(train_loader)}, Dx: {Dx:.5f}, DGz: {DGz:.5f}, D loss: {d_error:.5f}, G loss: {g_error:.5f}")

            summary_writer.add_scalars(
                "loss",
                {"D": d_error, "G": g_error},
                step
            )

            step += 1

        with torch.no_grad():
            fixed_output = generator.forward(fixed_noise).detach().cpu()
            fixed_noise_generator_output[i] = vutils.make_grid(fixed_output, padding=2, normalize=True)
            plt.imshow(np.transpose(fixed_noise_generator_output[i], (1,2,0)))
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(f"{str(log_dir)}/{i}.png",dpi=400,bbox_inches=0)


def get_summary_writer_log_dir(args):
    tb_log_dir_prefix = f'bs={args.batch_size}_lr={args.learning_rate}_run_'
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

if __name__ == '__main__':
    main(parser.parse_args())
