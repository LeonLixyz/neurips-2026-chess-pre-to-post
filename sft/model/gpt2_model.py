import torch
import torch.nn as nn
import math
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
try:
    from transformers.generation import GenerationMixin
except ImportError:
    from transformers import GenerationMixin
from .components import CausalSelfAttention, SwiGLUMLP, Transformer

class GPTConfig(PretrainedConfig):
  model_type = "custom_gpt"
  def __init__(self, vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embed=768, expansion_factor=4, bias=False, dropout=0.1, mlp_type="swiglu", **kwargs):
    super().__init__(**kwargs)
    self.vocab_size = vocab_size
    self.block_size = block_size
    self.n_layer = n_layer
    self.n_head = n_head
    self.n_embed = n_embed
    self.expansion_factor = expansion_factor
    self.bias = bias
    self.dropout = dropout
    self.mlp_type = mlp_type
    # Required for HuggingFace generation compatibility
    self.is_encoder_decoder = False
    self.is_decoder = True
    self.max_position_embeddings = block_size
    self.max_length = block_size
    
    # Standard HuggingFace attribute names (aliases)
    self.num_hidden_layers = n_layer
    self.num_attention_heads = n_head
    self.hidden_size = n_embed
    
    # Ensure use_return_dict is set (defaults to True)
    if not hasattr(self, 'use_return_dict'):
      self.use_return_dict = True

class GPT(PreTrainedModel, GenerationMixin):
  config_class = GPTConfig
  
  def __init__(self, config):
    super().__init__(config)
    self.token_emb = nn.Embedding(config.vocab_size, config.n_embed)
    self.pos_emb = nn.Embedding(config.block_size, config.n_embed)
    self.emb_dropout = nn.Dropout(config.dropout)
    self.blocks = nn.ModuleList([Transformer(config) for _ in range(config.n_layer)])
    self.ln_f = nn.LayerNorm(config.n_embed)
    self.head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
    self.block_size = config.block_size
    
    self.apply(self._init_weights)
    
    # Initialize the output projection layer with the scaled initialization (mentioned in the paper)
    for pn, p in self.named_parameters():
      if pn.endswith('out_proj.weight'):
          torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
    
    print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
  def _init_weights(self, module: nn.Module):
    if isinstance(module, nn.Linear):
      nn.init.xavier_uniform_(module.weight)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)
    # elif isinstance(module, nn.LayerNorm):
    #   nn.init.zeros_(module.bias)
    #   nn.init.ones_(module.weight)
  
  def get_num_params(self):
      n_params = sum(p.numel() for p in self.parameters())
      return n_params

  def forward(self, input_ids, targets=None, labels=None, past_key_values=None, use_cache=False, position_ids=None, attention_mask=None, return_dict=None, **kwargs):
    # Support both 'targets' (custom) and 'labels' (HuggingFace standard)
    if labels is not None and targets is None:
      targets = labels
    
    # Use return_dict if provided, otherwise use config default
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
      
    bsz, seqlen = input_ids.shape
    # Check if we have valid past_key_values
    if (past_key_values is None or 
        len(past_key_values) == 0 or 
        past_key_values[0] is None or
        (isinstance(past_key_values[0], (tuple, list)) and past_key_values[0][0] is None)):
      past_length = 0
      past_key_values = tuple([None] * self.config.n_layer)
    else:
      past_length = past_key_values[0][0].shape[2]
    if position_ids is None:
      position_ids = torch.arange(past_length, past_length + seqlen, device=input_ids.device).unsqueeze(0)
    
    # Ensure we don't exceed max position embeddings
    if past_length + seqlen > self.block_size:
      raise ValueError(f"Sequence length ({past_length + seqlen}) exceeds maximum position embeddings ({self.block_size})")
    x = self.token_emb(input_ids) + self.pos_emb(position_ids)
    x = self.emb_dropout(x)
    new_past_key_values = ()
    for i, block in enumerate(self.blocks):
      x, layer_cache = block(x, past_key_value=past_key_values[i], use_cache=use_cache)
      if use_cache:
        new_past_key_values += (layer_cache,)
    x = self.ln_f(x)
    logits = self.head(x)
    loss = None
    if targets is not None:
      loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    
    if not return_dict:
      output = (logits,) + (new_past_key_values,) if use_cache else (logits,)
      return ((loss,) + output) if loss is not None else output
    
    return CausalLMOutputWithPast(logits=logits, loss=loss, past_key_values=new_past_key_values if use_cache else None)

  def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
    """
    Prepare inputs for generation. This method is called by HuggingFace's generate().
    """
    # Only use the last token if we have valid past_key_values (KV cache)
    has_valid_cache = (
        past_key_values is not None and 
        len(past_key_values) > 0 and 
        past_key_values[0] is not None and
        (not isinstance(past_key_values[0], (tuple, list)) or past_key_values[0][0] is not None)
    )
    
    if has_valid_cache:
      input_ids = input_ids[:, -1:]
    
    return {
      "input_ids": input_ids,
      "past_key_values": past_key_values,
      "use_cache": kwargs.get("use_cache", True),
      "position_ids": kwargs.get("position_ids", None),
    }
  
  @staticmethod
  def _reorder_cache(past_key_values, beam_idx):
    """
    Reorder past_key_values cache for beam search.
    This method is called by HuggingFace's generate() when using beam search.
    """
    reordered_past = ()
    for layer_past in past_key_values:
      reordered_past += (
        tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
      )
    return reordered_past
  
  @property
  def device(self):
    """
    Return the device of the model (required for HuggingFace generation).
    """
    return next(self.parameters()).device
  
  def can_generate(self):
    """
    Returns whether the model can generate sequences.
    """
    return True
  
  def get_input_embeddings(self):
    """
    Returns the input embeddings layer.
    """
    return self.token_emb
  
  def set_input_embeddings(self, new_embeddings):
    """
    Sets the input embeddings layer.
    """
    self.token_emb = new_embeddings