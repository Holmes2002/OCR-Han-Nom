import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from fairseq.models.transformer import TransformerDecoder
class SimplifiedTransformerDecoder(nn.Module):
    def __init__(self, layers, fc):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.fc = fc

    def forward(self, prev_output_tokens,       # Target input tokens (shape: batch, seq_len)
                    encoder_out,           # Encoder outputs in dict format
                    src_lengths=None,                  # Optional: lengths of source sequences
                    features_only=False):

        for layer in self.layers:
            print(layer)
            # Pass arguments dynamically based on the layer's expected inputs
            prev_output_tokens = layer.forward(       # Target input tokens (shape: batch, seq_len)
                    encoder_out=encoder_out,           # Encoder outputs in dict format
                    # src_lengths=None,                  # Optional: lengths of source sequences
                    # features_only=False                # Set to True for raw features

            )
        output = prev_output_tokens.transpose(0, 1)
        return self.fc(output)

def create_decoder_with_roberta_weights(target_dictionary, args, pretrained_model="KoichiYasuoka/roberta-base-japanese-aozora-char"):
    """
    Creates a TransformerDecoder initialized with weights from a pre-trained Roberta model.
    
    Args:
        target_dictionary (fairseq.data.Dictionary): Target vocabulary dictionary.
        pretrained_model (str): Name or path of the pre-trained Roberta model.
    
    Returns:
        TransformerDecoder: Decoder with weights initialized from Roberta.
        AutoTokenizer: Tokenizer associated with the Roberta model.
    """
    # Load Roberta model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    roberta_model = AutoModelForMaskedLM.from_pretrained(pretrained_model)
    roberta_config = AutoConfig.from_pretrained(pretrained_model)
    
    # Define decoder parameters based on Roberta's configuration
    decoder_embed_dim = roberta_config.hidden_size

    # Initialize decoder embeddings using target dictionary
    decoder_embed_tokens = nn.Embedding(len(target_dictionary), decoder_embed_dim)

    # Create TransformerDecoder from Fairseq
    decoder = TransformerDecoder(
        args = args,
        dictionary=target_dictionary,
        embed_tokens=decoder_embed_tokens,
        no_encoder_attn=True,  # Enables cross-attention
    )

    # print(new_decoder_dict.keys())
    print('-'*100)
    new_state_dict =  {} 
    for k, v in roberta_model.state_dict().items():
        # General replacements
        k = k.replace('roberta.encoder.', 'layers.')
        k = k.replace('attention.self', 'self_attn')
        k = k.replace('attention.output.LayerNorm', 'self_attn_layer_norm')
        k = k.replace('attention.output.dense', 'self_attn.out_proj')
        k = k.replace('intermediate.dense', 'fc1')
        k = k.replace('output.dense', 'fc2')
        k = k.replace('output.LayerNorm', 'final_layer_norm')

        # Specific replacements for query, key, and value projections
        if 'query' in k:
            k = k.replace('query', 'q_proj')
        elif 'value' in k:
            k = k.replace('value', 'v_proj')
        elif 'key' in k:
            k = k.replace('key', 'k_proj')

        # Add the modified key and its value to the new state dictionary
        new_state_dict[k] = v
    new_state_dict = {k.replace('layers.layer', 'layers'):v for k,v in new_state_dict.items()}
    # print(new_state_dict.keys())
    missing_keys, unexpected_keys = decoder.load_state_dict(
            new_state_dict, strict=False
                )
    print(decoder)
    print('Missing_keys :', missing_keys)
    print('-'*100)
    print('Unexpected_keys :', unexpected_keys)
    # del decoder['embed_tokens'] 
    # del decoder['embed_positions'] 
    # del decoder['layernorm_embedding'] 
    # del decoder['dropout_module'] 
    
    return decoder, tokenizer
import argparse
def read_args_from_roberta(roberta_args: argparse.Namespace):
        # TODO: this would become easier if encoder/decoder where using a similar
        # TransformerConfig object
        args = argparse.Namespace(**vars(roberta_args))
        attr_map = [
            ("encoder_attention_heads", "decoder_attention_heads"),
            ("encoder_embed_dim", "decoder_embed_dim"),
            ("encoder_embed_dim", "decoder_output_dim"),
            ("encoder_normalize_before", "decoder_normalize_before"),
            ("encoder_layers_to_keep", "decoder_layers_to_keep"),
            ("encoder_ffn_embed_dim", "decoder_ffn_embed_dim"),
            ("encoder_layerdrop", "decoder_layerdrop"),
            ("encoder_layers", "decoder_layers"),
            ("encoder_learned_pos", "decoder_learned_pos"),
            # should this be set from here ?
            ("max_positions", "max_target_positions"),
        ]
        for k1, k2 in attr_map:
            setattr(args, k2, getattr(roberta_args, k1))

        args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
        args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
        args.share_decoder_input_output_embed = not roberta_args.untie_weights_roberta
        return args
def init_decoder(vocab):
    roberta_model = roberta = torch.hub.load('pytorch/fairseq:main', 'roberta.base')
    args = read_args_from_roberta(roberta_model.model.args)
    # assert False
    from fairseq.data import Dictionary

    # Example target dictionary
    target_dictionary = Dictionary()
    target_dictionary.add_symbol('<pad>')
    target_dictionary.add_symbol('<s>')
    target_dictionary.add_symbol('</s>')
    # Add more symbols as needed...
    for char in vocab:
        target_dictionary.add_symbol(char)
    # Create the decoder with Roberta weights
    decoder, tokenizer = create_decoder_with_roberta_weights(target_dictionary,args)
    tokenizer = tokenizer.train_new_from_iterator(vocab, vocab_size=len(vocab))
    # decoder = SimplifiedTransformerDecoder(decoder.layers, decoder.output_projection)
    return decoder, tokenizer
    # Test the decoder
    # print(decoder)
    # print(f"Tokenizer: {tokenizer}")
if __name__ == '__main__':
    HanNom_vocab = open('/home1/vudinh/NomNaOCR/HanNom_Vocab/unique_vocab.txt').read().splitlines()
    model, _ = init_decoder(HanNom_vocab)
    model(None, None)