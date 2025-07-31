
import torch
import torch.nn.functional as F
from torch import nn as nn
import pytorch_lightning as pl
from .embedding import BERTEmbedding, SimpleEmbedding
from .bert4rec import BERT
from .diff4rec import DIFF
import math

import pdb

class LDM(pl.LightningModule):
    def __init__(self,
                 max_len: int = None,
                 num_items: int = None,
                 n_layer: int = None,
                 n_head: int = None,
                 n_b: int = None,
                 d_model: int = None,
                 dropout: float = .0,
                 barope: bool = False,
                 diff_steps: int = None,
                 norm_weight: float = 0.0,
                 skip_steps: int = None,
                 cf_weight: float = 0.0,
                 n_e_sh: int = None,
                 n_e_sp: int = None,
                 n_dit_layer: int = None,
    ):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.num_items = num_items
        self.n_b = n_b
        self.skip_steps = skip_steps
        self.cf_weight = cf_weight

        vocab_size = num_items + 2
        self.embedding = SimpleEmbedding(vocab_size=vocab_size, embed_size=d_model, dropout=dropout)

        b_vocab_size = n_b + 2
        self.b_embedding = SimpleEmbedding(vocab_size=b_vocab_size, embed_size=d_model, dropout=dropout)

        self.diff_steps = diff_steps

        self.bert = BERT(
            max_len = max_len,
            num_items = num_items,
            n_layer = n_layer,
            n_head = n_head,
            n_b = n_b,
            d_model = d_model,
            dropout = dropout,
            barope = barope,
        )

        self.diff = DIFF(
            d_model = d_model,
            n_b = n_b,
            diff_steps = diff_steps,
            norm_weight = norm_weight,
            n_e_sh = n_e_sh,
            n_e_sp = n_e_sp,
            n_dit_layer = n_dit_layer,
        )

    def bert_encode(self, x_seq, b_seq):
        mask = x_seq > 0
        x_emb = self.embedding(x_seq)
        b_emb = self.b_embedding(b_seq)
        x = self.bert(x_emb, x_seq, b_emb, b_seq, mask)
        return x

    def bert_encode_mask(self, x_seq, b_seq, labels):
        mask = x_seq > 0
        index = labels > 0

        random_mask = torch.rand(index.shape) < 0.1
        b_seq[random_mask] = self.n_b + 1

        x_emb = self.embedding(x_seq)
        b_emb = self.b_embedding(b_seq)
        x = self.bert(x_emb, x_seq, b_emb, b_seq, mask)
        return x

    def diffusion_forward_behavior(self, x_seq, b_seq, labels):
        mask = x_seq > 0
        index = labels > 0

        x_emb = self.embedding(x_seq)
        b_emb = self.b_embedding(b_seq)
        x = self.bert(x_emb, x_seq, b_emb, b_seq, mask)

        b_seq_mask = b_seq.clone()
        b_seq_mask[index] = self.n_b + 1
        b_mask_emb = self.b_embedding(b_seq_mask)

        x_mask = self.bert(x_emb, x_seq, b_mask_emb, b_seq_mask, mask)

        x_target = x[index]
        x_unified = x_mask[index]
        x_behavior_info = x_target

        t = torch.randint(0, self.diff_steps, (x_target.size(0),), device=x_target.device).long()
        x_target_noise = self.diff.q_sample(x_behavior_info, t)

        b_guide = b_emb[index]
        b_guide_seq = b_seq[index]

        # use classifier-free guidance
        if self.cf_weight > 0:
            classfier_free_mask = torch.rand(b_guide.size(0), device=b_guide.device) < 0.2
            b_guide[classfier_free_mask] = torch.zeros((torch.sum(classfier_free_mask), b_guide.size(-1)),
                                                       device=b_guide.device)
            b_guide_seq[classfier_free_mask] = 0
            
        pred_x_target = self.diff.denoise(x_target_noise, b_guide, t, b_guide_seq, x_unified)
        loss = self.diff.mse(pred_x_target, x_target)

        if self.diff.norm_weight > 0:
            x_emb_norm = torch.norm(x_target, dim=-1)
            pred_x_norm = torch.norm(pred_x_target, dim=-1)
            loss_norm = (x_emb_norm - pred_x_norm).pow(2).mean()

            loss += self.diff.norm_weight * loss_norm

        return pred_x_target, loss

    def diffusion_encode_behavior(self, x_seq, b_seq, labels=None, index=None):
        mask = x_seq > 0
        if index is None:
            index = labels > 0

        x_emb = self.embedding(x_seq)
        b_emb = self.b_embedding(b_seq)

        b_seq_mask = b_seq.clone()
        b_seq_mask[index] = self.n_b + 1
        b_mask_emb = self.b_embedding(b_seq)

        x_mask = self.bert(x_emb, x_seq, b_mask_emb, b_seq_mask, mask)

        x_unified = x_mask[index]
        x_target = torch.randn_like(x_unified)

        b_guide = b_emb[index]
        b_guide_seq = b_seq[index]

        x_target = self.diff.sample_ddim(b_guide, skip_steps=self.skip_steps, x=x_target, b_guide_seq=b_guide_seq, x_unified=x_unified)

        if self.cf_weight > 0:
            b_guide_free = torch.zeros((b_guide.size(0), b_guide.size(-1)), device=b_guide.device)
            b_guide_seq = torch.zeros_like(b_guide_seq)
            x_target_free = self.diff.sample_ddim(b_guide_free, skip_steps=self.skip_steps, x=x_target, b_guide_seq=b_guide_seq, x_unified=x_unified)

            x_target = (1 + self.cf_weight) * x_target + self.cf_weight * x_target_free

        return x_target
