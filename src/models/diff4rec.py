
import torch
import torch.nn.functional as F
from torch import nn as nn
import pytorch_lightning as pl
from .heads import BehaviorMoE
import math

import pdb

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas

class AdaLN_MoE(nn.Module):
    def __init__(self, d_model, n_b, n_e_sh, n_e_sp):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm_2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.final_layer = BehaviorMoE(d_model, n_b=n_b, n_e_sh=n_e_sh, n_e_sp=n_e_sp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )

    def forward(self, x, conv, x_unified, b_guide_index=None):
        alpha_1, beta_1, gamma_1, alpha_2, beta_2, gamma_2 = self.adaLN_modulation(conv).chunk(6, dim=1)
        
        x_norm = self.norm_1(x) * (1 + gamma_1) + beta_1
        x = x + alpha_1 * self.mlp(torch.cat([x_norm, x_unified], dim=-1))

        x_norm = self.norm_2(x) * (1 + gamma_2) + beta_2
        x = x + alpha_2 * self.final_layer(x_norm, b_guide_index)
        return x

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DIFF(pl.LightningModule):
    def __init__(self, d_model, n_b, diff_steps, norm_weight, n_e_sh, n_e_sp, n_dit_layer):
        super().__init__()
        self.d_model = d_model
        self.diff_steps = diff_steps
        self.step_embedding = SinusoidalPositionEmbeddings(d_model)

        self.betas = exp_beta_schedule(timesteps=self.diff_steps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                    1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.denoiser = nn.ModuleList(
            [AdaLN_MoE(d_model, n_b, n_e_sh, n_e_sp) for _ in range(n_dit_layer)]
        )

        self.mse = nn.SmoothL1Loss(beta=1.0)

        self.norm_weight = norm_weight

    def denoise(self, x_t, h, t, b_seq, x_unified):
        t_emb = self.step_embedding(t)
        conv = h + t_emb
        x = x_t
        for layer in self.denoiser:
            x = layer(x, conv, x_unified, b_seq)
        return x

    def forward(self, x, h):
        batch_size = x.size(0)
        device = x.device

        t = torch.randint(0, self.diff_steps, (batch_size,), device=device).long()  # randomly sample t from 0 to diff_steps-1
        x_t = self.q_sample(x, t)

        pred_x_0 = self.denoise(x_t, h, t)

        loss = self.mse(pred_x_0, x)  # (B, d_model)

        if self.norm_weight > 0:
            x_emb_norm = torch.norm(x, dim=-1)  # [bs]
            pred_x_0_norm = torch.norm(pred_x_0, dim=-1)
            loss_norm = (x_emb_norm - pred_x_0_norm).pow(2).mean()

            loss += self.norm_weight * loss_norm

        return pred_x_0, loss

    def q_sample(self, x, t):
        noise = torch.randn_like(x)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        x_t = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t

    def p_sample_ddim(self, x_t, t_origin, bt_emb, skip_steps, b_guide_seq, x_unified):
        t = torch.full((x_t.shape[0],), t_origin, device=x_t.device, dtype=torch.long).long()  # [bs]
        next_t = t - skip_steps

        pred_x_0 = self.denoise(x_t, bt_emb, t, b_guide_seq, x_unified)

        if t_origin == 0:
            return pred_x_0

        else:
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
            sqrt_alphas_cumprod_next_t = extract(self.sqrt_alphas_cumprod, next_t, x_t.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            sqrt_one_minus_alphas_cumprod_next_t = extract(self.sqrt_one_minus_alphas_cumprod, next_t, x_t.shape)

            return (
                sqrt_alphas_cumprod_next_t * pred_x_0 +
                sqrt_one_minus_alphas_cumprod_next_t * (
                    (x_t - sqrt_alphas_cumprod_t * pred_x_0) / sqrt_one_minus_alphas_cumprod_t
                )
            )

    def sample_ddim(self, bt_emb, skip_steps, x=None, b_guide_seq=None, x_unified=None):
        if x is None:
            x = torch.randn_like(bt_emb)

        for t in list(range(0, self.diff_steps, skip_steps))[::-1]:
            x = self.p_sample_ddim(x, t, bt_emb, skip_steps, b_guide_seq, x_unified)

        return x
