import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUMLP(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.gate_value_proj = nn.Linear(config.n_embed, config.n_embed * config.expansion_factor * 2, bias=config.bias)
    self.out_proj = nn.Linear(config.n_embed * config.expansion_factor, config.n_embed, bias=config.bias)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x_proj = self.gate_value_proj(x)
    value, gate = x_proj.chunk(2, dim=-1)

    x = value * nn.functional.silu(gate)
    x = self.dropout(x)
    return self.out_proj(x)
  
class GELUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner_dim = config.n_embed * config.expansion_factor

        self.fc1 = nn.Linear(config.n_embed, inner_dim, bias=config.bias)
        self.fc2 = nn.Linear(inner_dim, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)  # or F.gelu(x, approximate="tanh") if you prefer
        x = self.dropout(x)
        x = self.fc2(x)
        return x