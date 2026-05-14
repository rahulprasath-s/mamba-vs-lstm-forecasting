import torch
import torch.nn as nn
from src.mamba_minimal import Mamba, ModelArgs


class MambaForecaster(nn.Module):
    def __init__(self, input_size=3, d_model=128, n_layers=2, output_steps=24):
        super().__init__()

        
        self.input_proj = nn.Linear(input_size, d_model)

        args = ModelArgs(
            d_model   = d_model,
            n_layer   = n_layers,
            vocab_size= d_model 
        )
        self.mamba = Mamba(args)

        self.fc = nn.Linear(d_model, output_steps)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.mamba(x)
        x = x[:, -1, :]
        return self.fc(x)