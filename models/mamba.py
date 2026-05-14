from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class _ResidualMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if Mamba is None:
            self.mixer = _FallbackSequenceMixer(d_model=d_model, dropout=dropout)
        else:
            self.mixer = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        return x + self.dropout(self.mixer(self.norm(x)))


class _FallbackSequenceMixer(nn.Module):

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=5,
            padding=2,
            groups=d_model,
        )
        self.gate = nn.Linear(d_model, d_model * 2)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        conv = self.depthwise(x.transpose(1, 2)).transpose(1, 2)
        value, gate = self.gate(x).chunk(2, dim=-1)
        mixed = F.silu(conv + value) * torch.sigmoid(gate)
        return self.out(self.dropout(mixed))


class MambaForecastModel(nn.Module):

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_size: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.blocks = nn.ModuleList(
            [
                _ResidualMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, -1])
