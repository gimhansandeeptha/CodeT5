from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

# Load the pre-trained CodeT5 model
model_name = "Salesforce/codet5-small"  # You can choose other variants like codet5-small, codet5-large, etc.
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Extract the decoder
decoder = model.decoder

# Example input: Let's assume we have encoded input and attention mask
encoded_input = torch.randint(0, 100, (1, 10))  # Example tensor, replace with real data
attention_mask = torch.ones_like(encoded_input)

# Initialize decoder inputs for sequence generation
decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]])  # Start with <pad> token
decoder_attention_mask = torch.ones_like(decoder_input_ids)

# Decode step-by-step or for sequence generation
outputs = decoder(
    input_ids=decoder_input_ids,
    attention_mask=decoder_attention_mask,
    encoder_hidden_states=torch.randn(1, 10, model.config.d_model),  # Example encoder hidden states
    encoder_attention_mask=attention_mask,
)

# Print shape of output (logits or hidden states)
print("Decoder output shape:", outputs.last_hidden_state.shape)

# For sequence generation, you'll need a loop or beam search method for autoregression.
