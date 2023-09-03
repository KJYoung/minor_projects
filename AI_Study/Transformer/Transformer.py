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

from Mask import make_src_mask, make_tgt_mask, make_src_tgt_mask

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
        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt)
        src_tgt_mask = make_src_tgt_mask(src, tgt)
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)
        out = self.generator(decoder_out)
        out = F.log_softmax(out, dim=-1)
        return out, decoder_out


from Encoding import TokenEmbedding, PositionalEncoding, TransformerEmbedding
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
):
    import copy

    copy = copy.deepcopy

    src_token_embed = TokenEmbedding(d_embed=d_embed, vocab_size=src_vocab_size)
    tgt_token_embed = TokenEmbedding(d_embed=d_embed, vocab_size=tgt_vocab_size)
    pos_embed = PositionalEncoding(d_embed=d_embed, max_len=max_len, device=device)

    src_embed = TransformerEmbedding(token_embed=src_token_embed, pos_embed=copy(pos_embed))
    tgt_embed = TransformerEmbedding(token_embed=tgt_token_embed, pos_embed=copy(pos_embed))

    attention = MultiHeadAttentionLayer(
        d_model=d_model, h=h, qkv_fc=nn.Linear(d_embed, d_model), out_fc=nn.Linear(d_model, d_embed)
    )
    position_ff = PositionWiseFeedForwardLayer(
        fc1=nn.Linear(d_embed, d_ff), fc2=nn.Linear(d_ff, d_embed)
    )

    encoder_block = EncoderBlock(self_attention=copy(attention), position_ff=copy(position_ff))
    decoder_block = DecoderBlock(
        self_attention=copy(attention),
        cross_attention=copy(attention),
        position_ff=copy(position_ff),
    )

    encoder = Encoder(encoder_block=encoder_block, n_layer=n_layer)
    decoder = Decoder(decoder_block=decoder_block, n_layer=n_layer)
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

