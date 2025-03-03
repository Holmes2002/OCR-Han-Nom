from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from vietocr.tool.translate import build_model, translate, translate_beam_search, process_input, predict
import torch 
from transformers import VisionEncoderDecoderModel, AutoModel
import torch.nn as nn 
from torch.nn import CrossEntropyLoss
import math
from typing import Optional, Tuple, Union

def vietocr(path_weight: str, vocab:str,  device: str):
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = path_weight
    config['cnn']['pretrained']=False
    # config['transformer']['num_decoder_layers'] = 8
    # config['transformer']['nhead'] = 16
    config['transformer']['d_model'] = 512
    config['predictor']['beamsearch']=False
    model, vocab = build_model(config, vocab)
    weights = config['weights']
    if weights != '':
        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))
    return model



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class TrVietOCR(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 width_size: int,
                 height_size: int,
                 max_length_token: int,
                 vietocr_pretrained: str,
                 decoder_vietocr_pretrained: str,
                 encoder_trocr_pretrained: str,
                 fc_vietocr_pretrained: str,
                 vocab = ''

                 ):
        ## load decoder
        super().__init__()
        self.device = "cuda"

        ## load encoder 
        self.encoder = VisionEncoderDecoderModel.from_pretrained(encoder_trocr_pretrained).encoder
        self.encoder.config.image_size = (384, 48)
        # assert False
        ## enc_to_dec_proj
        self.enc_to_dec_proj = nn.Linear(768, 512)
        self.embed_tgt = nn.Embedding(vocab_size + 5, 512)
        self.pos_enc = PositionalEncoding(512, 0.1, max_length_token)

        ## load fc 
        vietocr_model = vietocr(vietocr_pretrained, vocab, device=self.device)
        self.decoder = vietocr_model.transformer.transformer.decoder
        if decoder_vietocr_pretrained != '':
            self.decoder.load_state_dict(torch.load(decoder_vietocr_pretrained, map_location=torch.device(self.device)))
        self.fc = vietocr_model.transformer.fc 
        if fc_vietocr_pretrained != '':
            self.fc.load_state_dict(torch.load(fc_vietocr_pretrained, map_location=torch.device(self.device)))

        self.config = self.encoder.config
        self.config.decoder_start_token_id = 1
        self.config.pad_token_id = 0
        """
            ViTConfig {
            "_name_or_path": "./tmpencoder",
            "architectures": [
                "ViTModel"
            ],
            "attention_probs_dropout_prob": 0.0,
            "begin_suppress_tokens": null,
            "decoder_start_token_id": 2,
            "encoder_stride": 16,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "hidden_size": 768,
            "image_size": 384,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "model_type": "vit",
            "num_attention_heads": 12,
            "num_channels": 3,
            "num_hidden_layers": 12,
            "pad_token_id": 1,
            "patch_size": 16,
            "qkv_bias": false,
            "suppress_tokens": null,
            "tf_legacy_loss": false,
            "torch_dtype": "float32",
            "transformers_version": "4.19.1"
            }
        """

    def forward(self, 
                img=None,
                tgt_input=None,
                tgt_padding_mask=None,
                tgt_output = None
                ):
        encoder_outputs = self.encoder(
            img,
            output_attentions=self.config.output_attentions,
            output_hidden_states=self.config.output_hidden_states,
            return_dict=self.config.return_dict
        )  
        encoder_hidden_states = encoder_outputs[0]
        # print(encoder_hidden_states.shape)
        encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.transpose(1,0)
        tgt_mask = self.gen_nopeek_mask(tgt_input.shape[0]).to(self.device)
        tgt = self.pos_enc(self.embed_tgt(tgt_input) * math.sqrt(512))

        decoder_outputs = self.decoder(
            tgt=tgt,
            memory=encoder_hidden_states,
            tgt_mask = tgt_mask, 
            tgt_key_padding_mask = tgt_padding_mask
        )
        decoder_outputs= decoder_outputs.transpose(0, 1)
        decoder_outputs = self.fc(decoder_outputs)
        return decoder_outputs

    def gen_nopeek_mask(self, length):
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward_decoder(self, tgt, memory):
        tgt_mask = self.gen_nopeek_mask(tgt.shape[0]).to(tgt.device)
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(512))
        # tgt = tgt.transpose(1,0)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)
        return self.fc(output), memory
            

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

if __name__ == "__main__":
    model = TrVietOCR("cuda")
    print(model)