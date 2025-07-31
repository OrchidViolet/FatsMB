

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class DotProductPredictionHead(nn.Module):
    """share embedding parameters"""
    def __init__(self, d_model, num_items, token_embeddings):
        super().__init__()
        self.token_embeddings = token_embeddings
        self.vocab_size = num_items + 1
        self.out = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model),
            )
        self.bias = nn.Parameter(torch.zeros(1, self.vocab_size))

    def forward(self, x, b_seq, candidates=None):
        x = self.out(x)  # B x H or M x H
        if candidates is not None:  # x : B x H
            emb = self.token_embeddings(candidates)  # B x C x H
            logits = (x.unsqueeze(1) * emb).sum(-1)  # B x C
            bias = self.bias.expand(logits.size(0), -1).gather(1, candidates)  # B x C
            logits += bias
        else:  # x : M x H
            emb = self.token_embeddings.weight[:self.vocab_size]  # V x H
            logits = torch.matmul(x, emb.transpose(0, 1))  # M x V
            logits += self.bias
        return logits

class BehaviorMoE(nn.Module):
    """
    model with shared expert and behavior specific expert
    3 shared expert,
    1 specific expert per behavior.
    """

    def __init__(self, d_model, n_b, n_e_sh, n_e_sp):
        super().__init__()
        self.n_b = n_b
        self.n_e_sh = n_e_sh
        self.n_e_sp = n_e_sp
        self.softmax = nn.Softmax(dim=-1)
        self.shared_experts = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model)) for i in range(self.n_e_sh)])
        self.specific_experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_model, d_model)) for i in range(self.n_b * self.n_e_sp)])
        self.w_gates = nn.Parameter(torch.randn(self.n_b, d_model, self.n_e_sh + self.n_e_sp), requires_grad=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, b_seq):
        shared_experts_o = [e(x) for e in self.shared_experts]
        specific_experts_o = [e(x) for e in self.specific_experts]
        
        logits = torch.einsum('nd,tde->tne', x, self.w_gates)
        private_mask = (b_seq != 0)
        private_mask = private_mask.view(1,-1,1)

        mask = torch.ones(1,private_mask.size(1),self.n_e_sh + self.n_e_sp, dtype=torch.bool, device=logits.device)
        mask[:,:,self.n_e_sh:] = private_mask
        mask = mask.expand(logits.size(0), -1, -1)
        logits = logits.masked_fill(~mask, float('-inf'))

        gates_o = self.softmax(logits)

        experts_o_tensor = torch.stack(
            [torch.stack(shared_experts_o + specific_experts_o[i * self.n_e_sp:(i + 1) * self.n_e_sp]) for i in
             range(self.n_b)])

        output = torch.einsum('tend,tne->tnd', experts_o_tensor, gates_o)
        outputs = torch.cat([torch.zeros_like(x).unsqueeze(0), output])
        x = x + self.ln(torch.einsum('tnd, nt -> nd', outputs, F.one_hot(b_seq, num_classes=self.n_b + 1).float()))
        return x
