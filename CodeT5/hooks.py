
self_attention_inputs = {}
cross_attention_inputs = {}

def self_attention_hook(module, input, output):
    self_attention_inputs[f"block_{len(self_attention_inputs)}-self_attention_inputs"] = input

def cross_attention_hook(module, input, output):
    cross_attention_inputs[f"block_{len(cross_attention_inputs)}-cross_attention_inputs"] = input

def print_attention_inputs ():
    print("self_attention_inputs: ", self_attention_inputs.keys())
    print("cross_attention_inputs: ", cross_attention_inputs.keys())
    