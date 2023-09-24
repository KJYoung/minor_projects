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
    'b0': (0, 224, 0.2),
    'b1': (0.5, 240, 0.2),
    'b2': (1, 260, 0.3),
    'b3': (2, 300, 0.3),
    'b4': (3, 380, 0.4),
    'b5': (4, 456, 0.4),
    'b6': (5, 528, 0.5),
    'b7': (6, 600, 0.5),
}


class CNNBlock(nn.Module):
    pass


class SqueezeExcitation(nn.Module):
    pass


class InvertedResidualBlock(nn.Module):
    pass


class EfficientNet(nn.Module):
    pass
