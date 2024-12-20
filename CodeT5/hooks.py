from transformers.models.t5.modeling_t5 import T5Attention
EXTERNAL_DECODER = True

class AttentionInputsManager:
    def __init__(self):
        self.self_attention_inputs = {}
        self.cross_attention_inputs = {}

    def add_attention_input(self, attention_type, block_index, input_data):
        if attention_type == "self_attention":
            self.self_attention_inputs[f"block_{block_index}"] = input_data
        elif attention_type == "cross_attention":
            self.cross_attention_inputs[f"block_{block_index}"] = input_data

    def get_attention_inputs(self):
        return self.self_attention_inputs, self.cross_attention_inputs

    def clear_attention_inputs(self):
        self.self_attention_inputs.clear()
        self.cross_attention_inputs.clear()

# Hook functions that interact with the manager
def _self_attention_hook(manager):
    def hook(module, input, output):
        block_index = len(manager.self_attention_inputs)
        manager.add_attention_input("self_attention", block_index, input)
    return hook

def _cross_attention_hook(manager):
    def hook(module, input, output):
        block_index = len(manager.cross_attention_inputs)
        manager.add_attention_input("cross_attention", block_index, input)
    return hook

# Register hooks
def register_hooks(model, manager):
    if EXTERNAL_DECODER:
        for n in range(model.config.num_decoder_layers):
            model.decoder.block[n].layer[0].SelfAttention.register_forward_hook(_self_attention_hook(manager))
            model.decoder.block[n].layer[1].EncDecAttention.register_forward_hook(_cross_attention_hook(manager))

def external_decoder_hook(attention_object: T5Attention, mutable_key_value_obj):
    """
    Hook function to merge attention outputs with external key-value states.
    """
    print("external_decoder_hook registration function")
    def post_hook(module, input, output):
        # Compute external attention output
        attention_output = attention_object(
            hidden_states=input[0],
            key_value_states=mutable_key_value_obj,
            query_length=input[0].shape[1]
        )

        # Merge the outputs
        merged_output = (output[0] + attention_output[0]) / 2

        # Return a new tuple with the merged output replacing the original
        return (merged_output,) + output[1:]
    
    return post_hook
# self_attention_inputs = {}
# cross_attention_inputs = {}
# def self_attention_hook(module, input, output):
#     self_attention_inputs[f"block_{len(self_attention_inputs)}-self_attention_inputs"] = input

# def cross_attention_hook(module, input, output):
#     cross_attention_inputs[f"block_{len(cross_attention_inputs)}-cross_attention_inputs"] = input

# def delete_attention_inputs():
#     global self_attention_inputs, cross_attention_inputs
#     self_attention_inputs = {}
#     cross_attention_inputs = {}

# def get_attention_inputs():
#     return self_attention_inputs