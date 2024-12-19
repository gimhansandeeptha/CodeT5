import torch
import torch.nn as nn

class ModifiedDecoderLayer(nn.Module):
    def __init__(self, original_layer, d_model, num_heads, merge_method='average'):
        super(ModifiedDecoderLayer, self).__init__()
        # Existing components from CodeT5
        self.self_attention = original_layer.self_attn
        self.cross_attention = original_layer.cross_attn
        self.feed_forward = original_layer.linear2  # FFN layer

        # Additional attention blocks
        self.additional_attention1 = nn.MultiheadAttention(d_model, num_heads, dropout=original_layer.dropout.p)
        self.additional_attention2 = nn.MultiheadAttention(d_model, num_heads, dropout=original_layer.dropout.p)

        # Merge method: 'average', 'sum', or 'concat'
        self.merge_method = merge_method
        if merge_method == 'concat':
            self.merge_linear = nn.Linear(2 * d_model, d_model)

        # Normalization layers
        self.norm1 = original_layer.norm1
        self.norm2 = original_layer.norm2
        self.norm3 = original_layer.norm3

        # Dropout
        self.dropout = original_layer.dropout

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self-Attention
        tgt2 = self.self_attention(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        # Cross-Attention
        tgt2 = self.cross_attention(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]

        # Additional Attention Blocks
        additional1 = self.additional_attention1(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        additional2 = self.additional_attention2(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]

        # Merge the results
        if self.merge_method == 'average':
            tgt2 = (tgt2 + additional1 + additional2) / 3
        elif self.merge_method == 'sum':
            tgt2 = tgt2 + additional1 + additional2
        elif self.merge_method == 'concat':
            tgt2 = self.merge_linear(torch.cat((additional1, additional2), dim=-1))

        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        # Feed-Forward Network
        tgt2 = self.feed_forward(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class ModifiedDecoder(nn.Module):
    def __init__(self, original_decoder, d_model, num_heads, merge_method='average'):
        super(ModifiedDecoder, self).__init__()
        self.layers = nn.ModuleList([
            ModifiedDecoderLayer(layer, d_model, num_heads, merge_method)
            for layer in original_decoder.layers
        ])
        self.norm = original_decoder.norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return self.norm(tgt)

from transformers import T5ForConditionalGeneration

# Load pretrained CodeT5 model
codeT5 = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")

# Replace decoder with modified decoder
original_decoder = codeT5.model.decoder
d_model = original_decoder.config.d_model
num_heads = original_decoder.config.num_heads
modified_decoder = ModifiedDecoder(original_decoder, d_model, num_heads, merge_method='average')

# Assign modified decoder back to the model
codeT5.model.decoder = modified_decoder



import torch
from torch import nn
from transformers.models.t5.modeling_t5 import T5Attention, T5LayerNorm

class ModifiedT5DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = T5Attention(config, has_relative_attention_bias=True)
        self.cross_attn = T5Attention(config, is_decoder=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.ReLU(),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.additional_self_attn = T5Attention(config, has_relative_attention_bias=False)
        self.additional_cross_attn = T5Attention(config, is_decoder=True)
        
        self.layer_norm_self = T5LayerNorm(config.d_model)
        self.layer_norm_cross = T5LayerNorm(config.d_model)
        self.final_layer_norm = T5LayerNorm(config.d_model)

    def forward(
        self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None
    ):
        # Original Self-Attention
        normed_hidden_states = self.layer_norm_self(hidden_states)
        self_attn_output = self.self_attn(
            normed_hidden_states, mask=attention_mask, output_attentions=False
        )
        
        # Additional Self-Attention
        additional_self_attn_output = self.additional_self_attn(
            normed_hidden_states, mask=attention_mask, output_attentions=False
        )

        # Merge Self-Attention Outputs
        merged_self_attn_output = self_attn_output + additional_self_attn_output
        
        # Cross-Attention
        normed_hidden_states = self.layer_norm_cross(merged_self_attn_output)
        cross_attn_output = self.cross_attn(
            normed_hidden_states, key_value_states=encoder_hidden_states,
            mask=encoder_attention_mask, output_attentions=False
        )
        
        # Additional Cross-Attention
        additional_cross_attn_output = self.additional_cross_attn(
            normed_hidden_states, key_value_states=encoder_hidden_states,
            mask=encoder_attention_mask, output_attentions=False
        )

        # Merge Cross-Attention Outputs
        merged_cross_attn_output = cross_attn_output + additional_cross_attn_output
        
        # Feed Forward
        normed_hidden_states = self.final_layer_norm(merged_cross_attn_output)
        output = self.feed_forward(normed_hidden_states)
        
        return output



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# def scaled_dot_product(q, k, v, mask=None):
#     d_k = q.size()[-1]
#     scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
#     if mask is not None:
#         scaled = scaled.permute(1, 0, 2, 3) + mask
#         scaled = scaled.permute(1, 0, 2, 3)
#     attention = F.softmax(scaled, dim=-1)
#     values = torch.matmul(attention, v)
#     return values, attention

# class PositionwiseFeedForward(nn.Module):
#     def __init__(self, d_model, hidden, drop_prob=0.1):
#         super(PositionwiseFeedForward, self).__init__()
#         self.linear1 = nn.Linear(d_model, hidden)
#         self.linear2 = nn.Linear(hidden, d_model)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=drop_prob)

#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.linear2(x)
#         return x

# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads):
#         super().__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.head_dim = d_model // num_heads
#         self.qkv_layer = nn.Linear(d_model , 3 * d_model)
#         self.linear_layer = nn.Linear(d_model, d_model)

#     def forward(self, x, mask):
#         batch_size, sequence_length, d_model = x.size()
#         qkv = self.qkv_layer(x)
#         qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
#         qkv = qkv.permute(0, 2, 1, 3)
#         q, k, v = qkv.chunk(3, dim=-1)
#         values, attention = scaled_dot_product(q, k, v, mask)
#         values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
#         out = self.linear_layer(values)
#         return out

# class LayerNormalization(nn.Module):
#     def __init__(self, parameters_shape, eps=1e-5):
#         super().__init__()
#         self.parameters_shape=parameters_shape
#         self.eps=eps
#         self.gamma = nn.Parameter(torch.ones(parameters_shape))
#         self.beta =  nn.Parameter(torch.zeros(parameters_shape))

#     def forward(self, inputs):
#         dims = [-(i + 1) for i in range(len(self.parameters_shape))]
#         mean = inputs.mean(dim=dims, keepdim=True)
#         var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
#         std = (var + self.eps).sqrt()
#         y = (inputs - mean) / std
#         out = self.gamma * y + self.beta
#         return out
    
# class MultiHeadCrossAttention(nn.Module):
#     def __init__(self, d_model, num_heads):
#         super().__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.head_dim = d_model // num_heads
#         self.kv_layer = nn.Linear(d_model , 2 * d_model)
#         self.q_layer = nn.Linear(d_model , d_model)
#         self.linear_layer = nn.Linear(d_model, d_model)

#     def forward(self, x, y, mask):
#         batch_size, sequence_length, d_model = x.size()
#         kv = self.kv_layer(x)
#         q = self.q_layer(y)
#         kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)
#         # print("q_shape_0", q.shape)
#         q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
#         # print("q_shape_1",q.shape)
#         kv = kv.permute(0, 2, 1, 3)
#         q = q.permute(0, 2, 1, 3)
#         # print("q_shape_2", q.shape)
#         k, v = kv.chunk(2, dim=-1)
#         values, attention = scaled_dot_product(q, k, v, mask) # We don't need the mask for cross attention.
#         values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, d_model)
#         out = self.linear_layer(values)
#         return out

# class DecoderLayer(nn.Module):
#     def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
#         super(DecoderLayer, self).__init__()
#         self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
#         self.self_cross_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
#         self.layer_norm1 = LayerNormalization(parameters_shape=[d_model])
#         self.dropout1 = nn.Dropout(p=drop_prob)

#         self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
#         self.decoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
#         self.layer_norm2 = LayerNormalization(parameters_shape=[d_model])
#         self.dropout2 = nn.Dropout(p=drop_prob)

#         self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
#         self.layer_norm3 = LayerNormalization(parameters_shape=[d_model])
#         self.dropout3 = nn.Dropout(p=drop_prob)

#     # x - Encoder output
#     # y - second decoder main branch input
#     # self_x - Other decoder layers input
#     # cross_x - Other decoder's input to the MultiHeadAttention
#     def forward(self, x, y, self_x, cross_x, self_attention_mask, cross_attention_mask):
#         _y = y.clone()
#         y1 = self.self_attention(y, mask=self_attention_mask)
#         y2 = self.self_cross_attention(self_x, y, mask=cross_attention_mask)
#         # print(self_x.shape, y.shape)
#         y = (y1 + y2)/2
#         y = self.dropout1(y)
#         y = self.layer_norm1(y + _y)

#         _y = y.clone()
#         # print(x.shape, y.shape)
#         y1 = self.encoder_decoder_attention(x, y, mask=cross_attention_mask)
#         y2 = self.decoder_decoder_attention(cross_x, y, mask=cross_attention_mask)
#         y = (y1 + y2)/2
#         y = self.dropout2(y)
#         y = self.layer_norm2(y + _y)

#         _y = y.clone()
#         y = self.ffn(y)
#         y = self.dropout3(y)
#         y = self.layer_norm3(y + _y)
#         return y
    
# class CustomDecoder(nn.Module):
#     def __init__(self, num_layers, d_model, ffn_hidden, num_heads, drop_prob, vocab_size):
#         super(CustomDecoder, self).__init__()
#         self.layers = nn.ModuleList([
#             DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)
#         ])
#         self.linear = nn.Linear(d_model, vocab_size)

#     def forward(self, encoder_output, grouped_self_blocks, grouped_cross_blocks, self_attention_mask, cross_attention_mask):
#         # autoregressive
#         y = torch.zeros_like(grouped_self_blocks['iter_0'][0][0])
#         print(y.shape)
#         output_logits = []
#         for j in range (len(grouped_self_blocks)):
#             # each layer
#             for i, layer in enumerate(self.layers):
#                 self_x = grouped_self_blocks[f"iter_{j}"][i][0]
#                 cross_x = grouped_cross_blocks[f"iter_{j}"][i][0]

#                 y = layer(encoder_output, y, self_x, cross_x, self_attention_mask, cross_attention_mask)
#             logits = self.linear(y)
#             output_logits.append(logits)

#         output_logits = torch.cat(output_logits, dim=1)
#         return output_logits
