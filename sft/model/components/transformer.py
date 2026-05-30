import torch.nn as nn

from .attention import CausalSelfAttention
from .mlp import SwiGLUMLP, GELUMLP

class Transformer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.ln1 = nn.LayerNorm(config.n_embed, eps=1e-5)
    self.ln2 = nn.LayerNorm(config.n_embed, eps=1e-5)
    self.attn = CausalSelfAttention(config.n_embed, config.n_head, config.block_size, config.dropout)
    if config.mlp_type == "swiglu":
      self.mlp = SwiGLUMLP(config)
    elif config.mlp_type == "gelu":
      self.mlp = GELUMLP(config)
    else:
      raise ValueError(f"Invalid MLP type: {config.mlp_type}")

  def forward(self, x, past_key_value=None, use_cache=False):
    attn_output, cache = self.attn(self.ln1(x), past_key_value=past_key_value, use_cache=use_cache)
    x = x + attn_output
    x = x + self.mlp(self.ln2(x))
    return x, cache