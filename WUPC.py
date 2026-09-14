# -*- coding: utf-8 -*-
"""Weight-guided Unknown-speaker Prototype Constructor (WUPC).

A plug-and-play module that constructs a *negative prototype* for open-set
speaker recognition / few-shot open-set tasks. Given the ``n_way`` known-class
prototypes of one episode, WUPC synthesises a single embedding direction that
stands for "speakers belonging to none of the known classes", so that the
classifier can score unknown speakers on an explicit repulsion direction
instead of relying on a threshold over known-class posteriors.
"""

import os
import torch
import torch.nn as nn

__all__ = ["WUPC"]

def _load_memory(path, name):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "{} not found: {}\n"
            "Pass the path to the .pth produced by the pre-training stage.".format(name, path)
        )

    try:
        memory = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        memory = torch.load(path, map_location="cpu")

    if not torch.is_tensor(memory):
        raise TypeError("{} must be saved as a Tensor, got {}".format(name, type(memory)))
    if memory.dim() != 2:
        raise ValueError(
            "{} must be 2-D [num_entries, dim], got shape {}".format(name, tuple(memory.shape))
        )
    return memory.float()


class WUPC(nn.Module):
    def __init__(self, dim, base_speakers_weight_path, open_set_weight_path,
                 num_heads=1, mlp_hidden_dim=None, gate_reduction=4):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                "dim ({}) must be divisible by num_heads ({})".format(dim, num_heads)
            )
        hidden = dim // gate_reduction
        if hidden < 1:
            raise ValueError(
                "gate_reduction ({}) is too large for dim ({})".format(gate_reduction, dim)
            )

        self.dim = dim
        if mlp_hidden_dim is None:
            mlp_hidden_dim = dim

        self.register_buffer("base_speakers_weight",
                             _load_memory(base_speakers_weight_path, "base_speakers_weight"))
        self.register_buffer("open_set_weight",
                             _load_memory(open_set_weight_path, "open_set_weight"))
        self.context_norm = nn.LayerNorm(dim * 2)
        self.task_gate_A = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

        # ------------------------------------------------------------------
        # Stage A: Base space anchor block
        # ------------------------------------------------------------------
        self.attn_A = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_q_A = nn.LayerNorm(dim)
        self.norm_k_A = nn.LayerNorm(dim)
        self.norm_v_A = nn.LayerNorm(dim)
        self.mlp_A = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim * 2, dim),
        )

        # ------------------------------------------------------------------
        # Stage B: Open space projection block
        # ------------------------------------------------------------------
        self.attn_B = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_q_B = nn.LayerNorm(dim)
        self.norm_k_B = nn.LayerNorm(dim)
        self.norm_v_B = nn.LayerNorm(dim)

    def forward(self, prototypes, support_feat_grouped):
        if prototypes.dim() != 2:
            raise ValueError(
                "prototypes must be [n_way, dim], got {}".format(tuple(prototypes.shape))
            )
        if support_feat_grouped.dim() != 3:
            raise ValueError(
                "support_feat_grouped must be [n_way, k_shot, dim], got {}".format(
                    tuple(support_feat_grouped.shape)
                )
            )

        n_way, dim = prototypes.shape

        mean_context = prototypes.mean(dim=0, keepdim=True)  # [1, dim]
        std_context = torch.std(
            support_feat_grouped, dim=1, unbiased=False
        ).mean(dim=0, keepdim=True)  # [1, dim]

        context = torch.cat([mean_context, std_context], dim=-1)
        context = self.context_norm(context)

        gate_A = self.task_gate_A(context).unsqueeze(1)

        q_A = self.norm_q_A(prototypes.unsqueeze(0))
        base = self.base_speakers_weight.unsqueeze(0)

        k_A = self.norm_k_A(base) * gate_A
        v_A = self.norm_v_A(base)

        attn_out_A, _ = self.attn_A(query=q_A, key=k_A, value=v_A)

        pooled_A = attn_out_A.mean(dim=1).squeeze(0)
        Cn = self.mlp_A(pooled_A)

        # ==================================================================
        # Stage B: Open space projection block
        # ==================================================================
        q_B = self.norm_q_B(Cn.view(1, 1, dim))
        open_bank = self.open_set_weight.unsqueeze(0)
        k_B = self.norm_k_B(open_bank)
        v_B = self.norm_v_B(open_bank)

        attn_out_B, _ = self.attn_B(query=q_B, key=k_B, value=v_B)
        residual_B = Cn.view(1, 1, dim) + attn_out_B
        neg_proto = residual_B.mean(dim=1).squeeze(0)

        return neg_proto
