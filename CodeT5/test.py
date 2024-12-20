import torh
from t5_attention import AttentionModule, MutableKeyValueStates
from external_decoder import ExternalDecoder
from hooks import external_decoder_hook

batch_size = 1
seq_length = 16
d_model = 512

attentionModule = AttentionModule()
key_value_states = torch.randn(batch_size, seq_length, d_model)
mutable_key_value_states = MutableKeyValueStates()
mutable_key_value_states.__setitem__(key_value_states)

external_decoder = ExternalDecoder()
external_decoder.decoder.block[0].layer[0].SelfAttention.register_forward_hook(external_decoder_hook(attention_object=attentionModule, mutable_key_value_states=mutable_key_value_states))
external_decoder.forward()
