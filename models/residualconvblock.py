import torch
import torch.nn as nn
import torch.nn.functional as F


# Residual Convolutional Block Module for the Autoencoder and Constrastive model
class ResiudualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = F.relu(residual)

        residual = self.conv2(residual)
        residual = self.bn2(residual)

        skip = self.skip(x)

        out = residual + skip
        out = F.relu(out)

        return out