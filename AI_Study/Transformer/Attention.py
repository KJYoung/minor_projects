import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

# Singlehead attention
# def calculate_attention(query, key, value, mask):
#     # query, key, value: (n_batch, seq_len, d_k)
#     # mask: (n_batch, seq_len, seq_len)
#     d_k = key.shape[-1]
#     attention_score = torch.matmul(
#         query, key.transpose(-2, -1)
#     )  # Q x K^T, (n_batch, seq_len, seq_len)
#     attention_score = attention_score / math.sqrt(d_k)
#     if mask is not None:
#         attention_score = attention_score.masked_fill(mask == 0, -1e9)
#     attention_prob = F.softmax(attention_score, dim=-1)  # (n_batch, seq_len, seq_len)
#     out = torch.matmul(attention_prob, value)  # (n_batch, seq_len, d_k)
#     return out

# Multihead attention
def calculate_attention(self, query, key, value, mask):
    # query, key, value: (n_batch, h, seq_len, d_k)
    # mask: (n_batch, 1, seq_len, seq_len)
    d_k = key.shape[-1]
    attention_score = torch.matmul(
        query, key.transpose(-2, -1)
    )  # Q x K^T, (n_batch, h, seq_len, seq_len)
    attention_score = attention_score / math.sqrt(d_k)
    if mask is not None:
        attention_score = attention_score.masked_fill(mask == 0, -1e9)
    attention_prob = F.softmax(attention_score, dim=-1)  # (n_batch, h, seq_len, seq_len)
    out = torch.matmul(attention_prob, value)  # (n_batch, h, seq_len, d_k)
    return out


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, h, qkv_fc, out_fc):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = copy.deepcopy(qkv_fc)  # (d_embed, d_model)
        self.k_fc = copy.deepcopy(qkv_fc)  # (d_embed, d_model)
        self.v_fc = copy.deepcopy(qkv_fc)  # (d_embed, d_model)
        self.out_fc = out_fc  # (d_model, d_embed)

    def forward(self, query, key, value, mask=None):
        # query, key, value: (n_batch, seq_len, d_embed)
        # mask: (n_batch, seq_len, seq_len)
        # return value: (n_batch, h, seq_len, d_embed) # Typo fixed.
        n_batch = query.size(0)

        def transform(x, fc):  # (n_batch, seq_len, d_embed)
            out = fc(x)  # (n_batch, seq_len, d_model)
            out = out.view(
                n_batch, -1, self.h, self.d_model // self.h
            )  # (n_batch, seq_len, h, d_k)
            out = out.transpose(1, 2)  # (n_batch, h, seq_len, d_k)
            # out.transpose(1,2): calculate_attention에서 (n_batch, ..., seq_len, d_k)를 대상으로 연산을 수행하므로.
            return out

        query = transform(query, self.q_fc)  # (n_batch, h, seq_len, d_k)
        key = transform(key, self.k_fc)  # (n_batch, h, seq_len, d_k)
        value = transform(value, self.v_fc)  # (n_batch, h, seq_len, d_k)

        out = self.calculate_attention(query, key, value, mask)  # (n_batch, h, seq_len, d_k)
        out = out.transpose(1, 2)  # (n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model)  # (n_batch, seq_len, d_model)
        out = self.out_fc(out)  # (n_batch, seq_len, d_embed)
        return out
