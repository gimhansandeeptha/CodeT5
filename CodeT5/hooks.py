
self_attention_inputs = {}
cross_attention_inputs = {}
EXTERNAL_DECODER = True
def self_attention_hook(module, input, output):
    self_attention_inputs[f"block_{len(self_attention_inputs)}-self_attention_inputs"] = input

def cross_attention_hook(module, input, output):
    cross_attention_inputs[f"block_{len(cross_attention_inputs)}-cross_attention_inputs"] = input

def delete_attention_inputs():
    self_attention_inputs = {}
    cross_attention_inputs = {}

def register_hooks(model):
    if EXTERNAL_DECODER:
        for n in range (model.config.num_decoder_layers):
            self_attention_input = model.decoder.block[n].layer[0].SelfAttention.register_forward_hook(self_attention_hook)
            cross_attention_input = model.decoder.block[n].layer[1].EncDecAttention.register_forward_hook(cross_attention_hook)

def get_attention_inputs():
    return self_attention_inputs