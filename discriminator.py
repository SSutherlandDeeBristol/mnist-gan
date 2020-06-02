import math

import torch
from torch import nn
from torch.nn import functional as F


class Discriminator(nn.Module):
    def __init__(self, image_width, image_height, input_channels):
        super(Discriminator,self).__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.input_channels = input_channels

        self.conv1 = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=2)
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=self.conv1.out_channels,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=2
        )
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)

        self.conv3 = nn.Conv2d(
            in_channels=self.conv2.out_channels,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=2
        )
        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)

        self.fc1 = nn.Linear(self.conv3.out_channels * math.ceil(image_height/8) * math.ceil(image_width/8),10)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input),0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = torch.flatten(x,start_dim=1)
        x = F.leaky_relu(self.fc1(x),0.2)
        return x
