from torch.utils.checkpoint import checkpoint
import torch.nn as nn
import torch.nn.functional as F
import torch
import gc



D_MODEL = 256
N_LAYERS = 3
EXPAND = 2
D_CONV = 3
DROPOUT = 0.1
USE_GRAD_CHECKPOINTING = True


class SSMBlock(nn.Module):
    def __init__(self, d_model=D_MODEL, expand=EXPAND, d_conv=D_CONV, dropout=DROPOUT):
        super().__init__()
        d_inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
            bias=True,
        )
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        seq_len = x.shape[1]
        x = self.norm(x)
        x1, z = self.in_proj(x).chunk(2, dim=-1)

        x1 = self.conv1d(x1.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x1 = F.silu(x1)

        dt = F.softplus(self.dt_proj(x1))
        y = (x1 * self.D + dt * x1) * F.silu(z)
        return residual + self.dropout(self.out_proj(y))


class MambaForecaster(nn.Module):
    def __init__(self, input_size=3, d_model=D_MODEL, n_layers=N_LAYERS, output_steps=OUTPUT_STEPS):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.blocks = nn.ModuleList([SSMBlock(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, output_steps)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            if self.training and USE_GRAD_CHECKPOINTING:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm(x)
        return self.fc(x[:, -1, :])


def build_mamba():
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    model = MambaForecaster().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    allocated = torch.cuda.memory_allocated() / 1024**2 if DEVICE == 'cuda' else 0
    print('MambaForecaster built')
    print(f'   Parameters       : {total_params/1e6:.1f}M')
    print(f'   VRAM after model : {allocated:.0f}MB / {total_vram:.0f}MB')
    return model


print('MambaForecaster class defined')
print(f'   Config: d_model={D_MODEL}, layers={N_LAYERS}, microbatch={BATCH_SIZE}, accum={GRAD_ACCUM_STEPS}')
