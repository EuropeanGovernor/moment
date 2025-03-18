import torch
from torch import nn
class ForecastingHead(nn.Module):
    def __init__(
        self, head_nf: int = 768 * 64, forecast_horizon: int = 96, head_dropout: int = 0
    ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(head_nf, forecast_horizon)

    def forward(self, x, input_mask: torch.Tensor = None):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x