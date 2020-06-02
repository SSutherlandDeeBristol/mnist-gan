import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self, noise_vector_size, output_width, output_height, batch_size, mean=0.0, stddev=1.0):
        super(Generator, self).__init__()

        self.noise_vector_size = noise_vector_size
        self.output_width = output_width
        self.output_height = output_height
        self.batch_size = batch_size
        self.mean = mean
        self.stddev = stddev

        self.deconv1 = nn.ConvTranspose2d(
            in_channels=self.noise_vector_size,
            out_channels=128,
            kernel_size=5,
            stride=2
        )
        self.bn1 = nn.BatchNorm2d(self.deconv1.out_channels)

        self.deconv2 = nn.ConvTranspose2d(
            in_channels=self.deconv1.out_channels,
            out_channels=128,
            kernel_size=5,
            stride=2,
        )
        self.bn2 = nn.BatchNorm2d(self.deconv2.out_channels)

        self.deconv3 = nn.ConvTranspose2d(
            in_channels=self.deconv2.out_channels,
            out_channels=1,
            kernel_size=4,
            stride=2
        )
        self.bn3 = nn.BatchNorm2d(self.deconv3.out_channels)

        self.forward()

    def forward(self):
        noise = torch.randn(self.batch_size, self.noise_vector_size, 1, 1)

        x = F.relu(self.deconv1(noise))

        x = F.relu(self.deconv2(x))

        x = F.relu(self.deconv3(x))

        return x