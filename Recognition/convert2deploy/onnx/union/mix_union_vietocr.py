import torch.nn as nn 
from union.vit_model import VisionTransformer
import torch
from vietocr.tool.config import Cfg
from vietocr.tool.translate import build_model
import math

def vietocr(path_weight: str, device: str):
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = path_weight
    config['cnn']['pretrained']=False
    config['device'] = device
    config['predictor']['beamsearch']=False
    model, vocab = build_model(config)
    weights = config['weights']
    if weights != '':
        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))
    return model

class UnionVietOCR(nn.Module):
    def __init__(self,
                 device="cuda",
                 embed_dim_vit=384,
                 max_length_token=256,
                 img_width=128,
                 img_height=32,
                 patch_size=4,
                 vietocr_pretrained='./weights/transformerocr.pth',
                 decoder_vietocr_pretrained='./weights/vietocr/decoder.pt',
                 fc_vietocr_pretrained='./weights/vietocr/fc.pt',
                 union_pretrained='./union/pretrained/vit_finetune.pth'):
        super().__init__()

        self.device = device
        ##union vit
        img_size = (img_height,img_width)
        self.vit_model = VisionTransformer(img_size=img_size, pretrained=None, embed_dim=embed_dim_vit, patch_size=patch_size)
        if img_size == (32,128) and union_pretrained != '':
            print("[INFO] Load pretrained maerec Union14M")
            self.vit_model.load_state_dict(torch.load(union_pretrained, map_location=torch.device(self.device)))

        ##mapping enc && dec
        self.enc_to_dec_proj = nn.Linear(embed_dim_vit, 256)
        self.embed_tgt = nn.Embedding(embed_dim_vit, 256)
        self.pos_enc = PositionalEncoding(256, 0.1, max_length_token)

        ##vietocr decoder
        vietocr_model = vietocr(vietocr_pretrained, device=self.device)
        self.decoder = vietocr_model.transformer.transformer.decoder
        if decoder_vietocr_pretrained != '':
            self.decoder.load_state_dict(torch.load(decoder_vietocr_pretrained, map_location=torch.device(self.device)))

        ##vietocr fc
        self.fc = vietocr_model.transformer.fc 
        if fc_vietocr_pretrained != '':
            self.fc.load_state_dict(torch.load(fc_vietocr_pretrained, map_location=torch.device(self.device)))

    def forward(self,
                img: torch.Tensor,
                tgt_input=None,
                tgt_padding_mask=None,
                tgt_output = None):
        encoder_output = self.vit_model(img)
        encoder_output = self.enc_to_dec_proj(encoder_output)
        encoder_output = encoder_output.transpose(1,0)

        tgt_mask = self.gen_nopeek_mask(tgt_input.shape[0]).to(self.device)
        tgt = self.pos_enc(self.embed_tgt(tgt_input) * math.sqrt(256))

        decoder_outputs = self.decoder(
            tgt=tgt,
            memory=encoder_output,
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
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(256))
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