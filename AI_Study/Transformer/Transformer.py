# https://cpm0722.github.io/pytorch-implementation/transformer
# http://nlp.seas.harvard.edu/2018/04/03/attention.html

# 0. Import
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

seaborn.set_context(context="talk")

# 1. Transformer Definition
class Transformer(nn.Module):
    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator):
        super(Transformer, self).__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    # def encode(self, x):
    # def encode(self, src, src_mask):
    #     # out = self.encoder(x)
    #     out = self.encoder(src, src_mask)
    #     return out

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


    def make_subsequent_mask(self, query, key):
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


    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    # def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
    #     out = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)
    #     return out

    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(self.tgt_embed(tgt), encoder_out, tgt_mask, src_tgt_mask)

    # def forward(self, x, z):
    #     c = self.encode(x)
    #     y = self.decode(z, c)
    #     return y

    # def forward(self, src, tgt):
    #     src_mask = make_src_mask(src)
    #     tgt_mask = make_tgt_mask(tgt)
    #     src_tgt_mask = make_src_tgt_mask(src, tgt)
    #     encoder_out = self.encode(src, src_mask)
    #     # y = self.decode(tgt, encoder_out, tgt_mask)
    #     y = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)
    #     return y

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src_tgt_mask = self.make_src_tgt_mask(src, tgt)
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)
        out = self.generator(decoder_out)
        out = F.log_softmax(out, dim=-1)
        return out, decoder_out


from Embedding import TokenEmbedding, PositionalEncoding, TransformerEmbedding
from Attention import MultiHeadAttentionLayer
from FeedForward import PositionWiseFeedForwardLayer
from Encoder import EncoderBlock, Encoder
from Decoder import DecoderBlock, Decoder


def build_model(
    src_vocab_size,
    tgt_vocab_size,
    device=torch.device("cpu"),
    max_len=256,
    d_embed=512,
    n_layer=6,
    d_model=512,
    h=8,
    d_ff=2048,
    dr_rate=0.1,
    norm_eps=1e-5
):
    import copy

    copy = copy.deepcopy

    src_token_embed = TokenEmbedding(d_embed=d_embed, vocab_size=src_vocab_size)
    tgt_token_embed = TokenEmbedding(d_embed=d_embed, vocab_size=tgt_vocab_size)

    pos_embed = PositionalEncoding(d_embed=d_embed, max_len=max_len, device=device)
    src_embed = TransformerEmbedding(token_embed=src_token_embed, pos_embed=copy(pos_embed), dr_rate=dr_rate)
    tgt_embed = TransformerEmbedding(token_embed=tgt_token_embed, pos_embed=copy(pos_embed), dr_rate=dr_rate)

    attention = MultiHeadAttentionLayer(
        d_model=d_model, h=h, qkv_fc=nn.Linear(d_embed, d_model), out_fc=nn.Linear(d_model, d_embed), dr_rate=dr_rate
    )
    position_ff = PositionWiseFeedForwardLayer(
        fc1=nn.Linear(d_embed, d_ff), fc2=nn.Linear(d_ff, d_embed), dr_rate=dr_rate
    )
    norm = nn.LayerNorm(d_embed, eps=norm_eps)

    encoder_block = EncoderBlock(
        self_attention=copy(attention), 
        position_ff=copy(position_ff), 
        norm = copy(norm),
        dr_rate=dr_rate
    )
    decoder_block = DecoderBlock(
        self_attention=copy(attention),
        cross_attention=copy(attention),
        position_ff=copy(position_ff),
        norm=copy(norm),
        dr_rate=dr_rate
    )

    encoder = Encoder(encoder_block=encoder_block, n_layer=n_layer, norm=copy(norm))
    decoder = Decoder(decoder_block=decoder_block, n_layer=n_layer, norm=copy(norm))
    generator = nn.Linear(d_model, tgt_vocab_size)

    model = Transformer(
        src_embed=src_embed,
        tgt_embed=tgt_embed,
        encoder=encoder,
        decoder=decoder,
        generator=generator,
    ).to(device)

    model.device = device

    return model

