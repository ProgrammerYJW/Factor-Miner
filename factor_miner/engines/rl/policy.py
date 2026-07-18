"""RL策略网络: token嵌入 + 小型TransformerEncoder + actor/critic双头 (CPU友好)."""
from __future__ import annotations

import torch
import torch.nn as nn

NEG_INF = -1e9


class PolicyNet(nn.Module):
    def __init__(self, n_actions: int, max_len: int,
                 d_model: int = 128, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.tok_emb = nn.Embedding(n_actions + 1, d_model)   # +1: BOS
        self.pos_emb = nn.Embedding(max_len + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.actor = nn.Linear(d_model, n_actions)
        self.critic = nn.Linear(d_model, 1)
        self.bos = n_actions

    def forward(self, seqs: torch.Tensor, lengths: torch.Tensor,
                masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """seqs: (B,L) 已右移含BOS; lengths: (B,) 有效长度; masks: (B,A) 合法动作.

        返回 (masked_logits (B,A), value (B,))
        """
        B, L = seqs.shape
        pos = torch.arange(L, device=seqs.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(seqs) + self.pos_emb(pos)
        causal = torch.triu(torch.ones(L, L, device=seqs.device, dtype=torch.bool), 1)
        pad = torch.arange(L, device=seqs.device).unsqueeze(0) >= lengths.unsqueeze(1)
        h = self.encoder(x, mask=causal, src_key_padding_mask=pad)
        last = h[torch.arange(B, device=seqs.device), lengths - 1]     # 各样本末位
        logits = self.actor(last).masked_fill(~masks, NEG_INF)
        return logits, self.critic(last).squeeze(-1)
