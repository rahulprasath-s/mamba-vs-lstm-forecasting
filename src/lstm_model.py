import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=2,
                 output_steps=24, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout
        )

        self.fc = nn.Linear(hidden_size, output_steps)

    def forward(self, x):
        out, _ = self.lstm(x)
        last    = out[:, -1, :]
        pred    = self.fc(last)
        return pred