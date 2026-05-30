import numpy as np
import torch
import torch.nn as nn
import math

class CausalSelfAttention(nn.Module):
  def __init__(self, hidden_dim, n_head, block_size, dropout):
    super().__init__()
    assert hidden_dim % n_head == 0
    self.n_head = n_head

    self.key = nn.Linear(hidden_dim, hidden_dim)
    self.query = nn.Linear(hidden_dim, hidden_dim)
    self.value = nn.Linear(hidden_dim, hidden_dim)
    self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    self.attn_dropout = nn.Dropout(dropout)
    self.resid_dropout = nn.Dropout(dropout)

    self.register_buffer(
      "mask", 
      torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0)
    )

  def forward(self, x, past_key_value=None, use_cache=False):
    bsz, seqlen, hidden_dim = x.size()
    head_dim = hidden_dim // self.n_head
    keys = self.key(x).view(bsz, seqlen, self.n_head, head_dim).transpose(1, 2)
    queries = self.query(x).view(bsz, seqlen, self.n_head, head_dim).transpose(1, 2)
    values = self.value(x).view(bsz, seqlen, self.n_head, head_dim).transpose(1, 2)
    if past_key_value is not None:
      keys = torch.cat([past_key_value[0], keys], dim=2)
      values = torch.cat([past_key_value[1], values], dim=2)
    past_len = 0 if past_key_value is None else past_key_value[0].shape[2]
    total_len = past_len + seqlen
    scores = (queries @ keys.transpose(-2, -1)) / math.sqrt(head_dim)
    scores = scores.masked_fill(self.mask[:, :, past_len : past_len + seqlen, : total_len] == 0, float('-inf'))
    scores = torch.softmax(scores, dim=-1)
    scores = self.attn_dropout(scores)
    y = scores @ values
    y = y.transpose(1, 2).contiguous().view(bsz, seqlen, hidden_dim)
    output = self.resid_dropout(self.out_proj(y))
    new_past_key_value = (keys, values) if use_cache else None
    return output, new_past_key_value