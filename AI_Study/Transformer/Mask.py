import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn


def make_pad_mask(self, query, key, pad_idx=1):  # Type fixed
    # query: (n_batch, query_seq_len)
    # key: (n_batch, key_seq_len)
    query_seq_len, key_seq_len = query.size(1), key.size(1)

    key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len)
    # Padding이면 False(0), Otherwise, True(1).
    key_mask = key_mask.repeat(1, 1, query_seq_len, 1)  # (n_batch, 1, query_seq_len, key_seq_len)

    query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # (n_batch, 1, query_seq_len, 1)
    query_mask = query_mask.repeat(1, 1, 1, key_seq_len)  # (n_batch, 1, query_seq_len, key_seq_len)

    mask = key_mask & query_mask
    mask.requires_grad = False
    return mask


def make_src_mask(self, src):
    pad_mask = self.make_pad_mask(src, src)
    return pad_mask


def make_subsequent_mask(query, key):
    # query: (n_batch, query_seq_len)
    # key: (n_batch, key_seq_len)
    query_seq_len, key_seq_len = query.size(1), key.size(1)

    tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype(
        'uint8'
    )  # lower triangle without diagonal
    mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)
    return mask


# Example: query_seq_len = key_seq_len = 6
# [[1, 0, 0, 0, 0, 0],
#  [1, 1, 0, 0, 0, 0],
#  [1, 1, 1, 0, 0, 0],
#  [1, 1, 1, 1, 0, 0],
#  [1, 1, 1, 1, 1, 0],
#  [1, 1, 1, 1, 1, 1]]

# tgt_mask도 pad_mask는 해야 함!
def make_tgt_mask(self, tgt):
    pad_mask = self.make_pad_mask(tgt, tgt)
    seq_mask = self.make_subsequent_mask(tgt, tgt)
    mask = pad_mask & seq_mask
    return pad_mask & seq_mask


def make_src_tgt_mask(self, src, tgt):
    pad_mask = self.make_pad_mask(tgt, src)
    return pad_mask
