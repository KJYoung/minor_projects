import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

from ResidualConnection import ResidualConnectionLayer


class EncoderBlock(nn.Module):
    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer() for _ in range(2)]

    # Before Pad Masking
    # def forward(self, x):
    #     out = x
    #     out = self.self_attention(out)
    #     out = self.position_ff(out)
    #     return out

    # With Pad Masking
    def forward(self, src, src_mask):
        out = src
        # out = self.self_attention(query=out, key=out, value=out, mask=src_mask)
        # out = self.position_ff(out)
        out = self.residuals[0](
            out, lambda out: self.self_attention(query=out, key=out, value=out, mask=src_mask)
        )
        out = self.residuals[1](out, self.position_ff)
        return out


class Encoder(nn.Module):
    # encoder_block: Encoder Block 객체.
    # n_layer: Encoder Block의 개수
    def __init__(self, encoder_block, n_layer):
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_block))

    # Before Pad Masking
    # def forward(self, x):
    #     out = x
    #     for layer in self.layers:
    #         out = layer(out)
    #     return out

    # With Pad Masking
    def forward(self, src, src_mask):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)
        return out
