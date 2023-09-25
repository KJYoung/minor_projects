import torch
import torch.nn as nn
from math import ceil

base_model = [
    # MBConv expand_ratio, Channels, Layers Repeat, Stride, Kernel Size
    [1, 16, 1, 1, 3],  # State 2
    [6, 24, 2, 2, 3],  # State 3
    [6, 40, 2, 2, 5],  # State 4
    [6, 80, 3, 2, 3],  # State 5
    [6, 112, 3, 1, 5],  # State 6
    [6, 192, 4, 2, 5],  # State 7
    [6, 320, 1, 1, 3],  # State 8
    # Stride fixed with error referred in README.md
]

# alpha(depth, 1.2), beta(width, 1.1), gamma(resolution, 1.15)
phi_values = {
    # tuple of (phi_value, resolution, drop_rate)
    'b0': (0, 224, 0.2),  # Baseline => [phi = 0]
    'b1': (0.5, 240, 0.2),  # ex. 224 * (1.15)^(0.5) = 240
    'b2': (1, 260, 0.3),  # ex. 224 * (1.15) = 257.6
    'b3': (2, 300, 0.3),
    'b4': (3, 380, 0.4),
    'b5': (4, 456, 0.4),
    'b6': (5, 528, 0.5),
    'b7': (6, 600, 0.5),
}


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,  # depth-wise conv를 하기 위해서 groups 옵션을 둬야 함.
        )
        # groups = 1 : Normal Conv
        # groups = in_channels : Depthwise Conv.
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C * H * W => C * 1 * 1
            nn.Conv2d(in_channels=in_channels, out_channels=reduced_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),  # resulting shape: C * 1 * 1
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


# MBConv
class InvertedResidualBlock(nn.Module):
    def __init__(self):
        super(InvertedResidualBlock, self).__init__()
        pass

    def forward(self, x):
        pass


class EfficientNet(nn.Module):
    pass
