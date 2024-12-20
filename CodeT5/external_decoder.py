from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
from t5_attention import AttentionModule
from hooks import external_decoder_hook

class ExternalDecoder():
    def __init__(self):
        print("initiate the external decoder")
        model_name = "Salesforce/codet5-small" 
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.decoder = self.model.decoder
        self.self_level_attention = AttentionModule()
        self.cross_level_attention = AttentionModule()

    def register_hooks(self, block_number, self_level_key_value_obj, cross_level_key_value_obj):
        self.decoder.block[block_number].layer[0].SelfAttention.register_forward_hook(external_decoder_hook(
                attention_object=self.self_level_attention,
                mutable_key_value_obj=self_level_key_value_obj))
        self.decoder.block[block_number].layer[1].EncDecAttention.register_forward_hook(external_decoder_hook(
                attention_object=self.cross_level_attention, 
                mutable_key_value_obj=cross_level_key_value_obj))
    
    def forward(self):
        # Example input
        encoded_input = torch.randint(0, 100, (1, 10)) 
        attention_mask = torch.ones_like(encoded_input)

        decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id]])  #<pad> token
        decoder_attention_mask = torch.ones_like(decoder_input_ids)

        # Decode sequence generation
        outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=torch.randn(1, 10, self.model.config.d_model),
            encoder_attention_mask=attention_mask,
        )

        # print("Decoder output shape:", outputs.last_hidden_state.shape)
        return outputs

        # For sequence generation, you'll need a loop or beam search method for autoregression.
