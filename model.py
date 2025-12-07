# model.py
import torch
import torch.nn as nn


class MFMConv2d(nn.Module):
    """
    Max-Feature-Map Conv2d:
    Conv2d -> split channels -> elementwise max.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

    def forward(self, x):
        x = self.conv(x)
        out1, out2 = x.chunk(2, dim=1)
        return torch.max(out1, out2)


class LCNN(nn.Module):
    """
    Light CNN for LFCC features.
    Input: (B, 1, F, T)
    Output: logits (B, num_classes)
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()

        self.block1 = nn.Sequential(
            MFMConv2d(in_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block2 = nn.Sequential(
            MFMConv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block3 = nn.Sequential(
            MFMConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block4 = nn.Sequential(
            MFMConv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)           # (B, 128, 1, 1)
        x = x.view(x.size(0), -1) # (B, 128)
        logits = self.fc(x)       # (B, num_classes)
        return logits