
import torch
import torch.nn.functional as F
from torch import nn as nn
import pytorch_lightning as pl
from .embedding import BERTEmbedding, SimpleEmbedding, PositionalEmbedding
import math

import pdb


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout=0):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, flag):
        return self.norm(x + self.dropout(sublayer(x)))


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, n_b, d_model, dropout=0.1):
        super().__init__()
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.n_b = n_b

        self.linear_layers = nn.Parameter(torch.randn(3, d_model, n_head, self.d_k))
        self.linear_layers.data.normal_(mean=0.0, std=0.02)

        self.b_proj = nn.Parameter(torch.randn(d_model, n_head, self.d_k))

        self.dropout = nn.Dropout(dropout)


    def forward(self, query, key, value, b_emb=None, mask=None, barope=False):
        batch_size, seq_len = query.size(0), query.size(1)

        query, key, value = [torch.einsum("bnd, dhk->bhnk", x, self.linear_layers[l])
                             for l, x in zip(range(3), (query, key, value))]

        if barope:
            b_rotate = torch.einsum("bnd, dhk->bhnk", b_emb, self.b_proj)

            query = self.rope(query, b_rotate)
            key = self.rope(key, b_rotate)
        else:
            query = self.rope(query)
            key = self.rope(key)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            assert len(mask.shape) == 2
            mask = (mask[:,:,None] & mask[:,None,:]).unsqueeze(1)
            if attn_scores.dtype == torch.float16:
                attn_scores = attn_scores.masked_fill(mask == 0, -65500)
            else:
                attn_scores = attn_scores.masked_fill(mask == 0, -1e30)
        p_attn = self.dropout(nn.functional.softmax(attn_scores, dim=-1))

        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_k)

        return x

    def rope(self, x, b_rotate=None):
        batch_size, num_heads, seq_len, head_dim = x.shape
        assert head_dim % 2 == 0

        half_dim = head_dim // 2
        device = x.device

        freq_seq = torch.arange(half_dim, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))  # (half_dim,)

        pos = torch.arange(seq_len, device=device, dtype=torch.float32)

        sinusoid_inp = torch.einsum('i,j->ij', pos, inv_freq)

        sin = torch.sin(sinusoid_inp).unsqueeze(0).unsqueeze(0)
        cos = torch.cos(sinusoid_inp).unsqueeze(0).unsqueeze(0)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        x_rotated_even = x1 * cos - x2 * sin
        x_rotated_odd = x1 * sin + x2 * cos

        x_out = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        x_out = x_out.flatten(-2)

        if b_rotate is not None:
            x_out = x_out * b_rotate

        return x_out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, n_b, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(n_head=n_head, n_b=n_b, d_model=d_model, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model * 4, d_model),
        )

        self.input_sublayer = SublayerConnection(d_model=d_model, dropout=dropout)
        self.output_sublayer = SublayerConnection(d_model=d_model, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, b_seq, mask, barope=False):
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, b_seq, mask=mask, barope=barope), False)
        x = self.output_sublayer(x, lambda _x: self.feed_forward(_x), True)
        return self.dropout(x)


class BERT(pl.LightningModule):
    def __init__(self,
        max_len: int = None,
        num_items: int = None,
        n_layer: int = None,
        n_head: int = None,
        n_b: int = None,
        d_model: int = None,
        dropout: float = .0,
        barope: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_items = num_items
        self.n_b = n_b
        self.barope = barope

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head, n_b, dropout) for _ in range(n_layer)]
        )

        self.position = PositionalEmbedding(max_len=max_len, d_model=d_model)

    def forward(self, x_emb, x_seq, b_emb, b_seq, mask):
        if self.barope:
            x = x_emb
        else:
            x = x_emb + b_emb

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, b_emb, mask, self.barope)

        return x
