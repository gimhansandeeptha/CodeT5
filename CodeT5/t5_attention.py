import torch
import torch.nn as nn
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Attention


class AttentionModule(nn.Module):
    def __init__(self, d_model=512, d_kv=64, num_heads=8, dropout_rate=0.1, 
                 is_decoder=True, relative_attention_num_buckets=32, 
                 relative_attention_max_distance=128, has_relative_attention_bias=True):
        super(AttentionModule, self).__init__()
        """
        Initializes the T5Attention layer with the given configuration.
        """
        config = T5Config(
            d_model=d_model,
            d_kv=d_kv,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            is_decoder=is_decoder,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
        )
        self.attention_layer = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)

    def forward(self, hidden_states, key_value_states=None, mask=None, output_attentions=True, query_length=None):
        """
        Forward method to compute the attention outputs.

        Args:
            hidden_states (torch.Tensor): Input hidden states [batch_size, seq_length, d_model].
            key_value_states (torch.Tensor, optional): Cross-attention key-value states. Defaults to None.
            mask (torch.Tensor, optional): Attention mask. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to True.
            query_length (int, optional): Length of the query. Defaults to None.

        Returns:
            tuple: Attention output and additional outputs like attention weights.
        """
        return self.attention_layer(
            hidden_states=hidden_states,
            key_value_states=key_value_states,
            mask=mask,
            output_attentions=output_attentions,
            query_length=query_length
        )
    