from union.mix_union_vietocr import UnionVietOCR
from union.load_config import Cfg
from union.utils import Vocab
from torch.nn.functional import log_softmax, softmax
from torchvision import transforms
from PIL import Image
import torch
import numpy as np 
import torch.nn as nn 
import cv2
import math


config = Cfg.load_config_from_file("./univietocr.yml")
vocab = config.vocab
device = config.device
img_size = (config.width_size, config.height_size)
max_length_token = config.max_length_token

model = UnionVietOCR(max_length_token=config.max_length_token,
                     img_width=config.width_size,
                     img_height=config.height_size,
                     patch_size=config.patch_size,
                     vietocr_pretrained=config.vietocr_pretrained,
                     decoder_vietocr_pretrained=config.decoder_vietocr_pretrained,
                     fc_vietocr_pretrained=config.fc_vietocr_pretrained,
                     union_pretrained=config.union_pretrained)


dir_ckpt = "/home1/data/thaitran/Research/OCR/source/end2end_univietocr/vietocr/union/ckpt/epoch_10.pth"
model.load_state_dict(torch.load(dir_ckpt))
model.to(device)
model.eval()

encoder = model.vit_model
enc_to_dec_proj = model.enc_to_dec_proj
decoder = model.decoder
fc = model.fc
pos_enc = model.pos_enc
embed_tgt = model.embed_tgt


class EncoderModel(nn.Module):
    def __init__(self, encoder, enc_to_dec_proj):
        super().__init__()
        self.encoder = encoder
        self.enc_to_dec_proj = enc_to_dec_proj
        self.encoder.eval()
        self.enc_to_dec_proj.eval()
    
    def forward(self, pixel_values):
        encoder_outputs = self.encoder(pixel_values)
        encoder_hidden_states = self.enc_to_dec_proj(encoder_outputs)
        encoder_hidden_states = encoder_hidden_states.transpose(1,0)

        return encoder_hidden_states
    
encoder_merge = EncoderModel(encoder=encoder, enc_to_dec_proj=enc_to_dec_proj)
encoder_merge.eval()



class EncoderDecoderModel(nn.Module):
    def __init__(self, decoder, fc, pos_enc, embed_tgt, encoder, enc_to_dec_proj):
        super().__init__()
        self.decoder = decoder
        self.fc = fc
        self.decoder.eval()
        self.fc.eval()
        self.pos_enc = pos_enc
        self.embed_tgt = embed_tgt
        self.max_length =1 
        self.max_seq_length = 256
        self.eos_token =2

        self.encoder = encoder
        self.enc_to_dec_proj = enc_to_dec_proj
        self.encoder.eval()
        self.enc_to_dec_proj.eval()

    def forward(self, pixel_values):

        encoder_outputs = self.encoder(pixel_values)
        encoder_hidden_states = self.enc_to_dec_proj(encoder_outputs)
        encoder_hidden_states = encoder_hidden_states.transpose(1,0)
        translated_sentence = torch.tensor([[1]*pixel_values.shape[0]])

        while not torch.all(torch.any(translated_sentence.T == self.eos_token, dim=1)):
            tgt = torch.LongTensor(translated_sentence).to(device)
            mask = (torch.triu(torch.ones(tgt.shape[0], tgt.shape[0])) == 1).transpose(0, 1)
            tgt_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(256))
            output = self.decoder(tgt, encoder_hidden_states, tgt_mask=tgt_mask)
            output = output.transpose(0, 1)
            output = self.fc(output)
            output = torch.softmax(output, dim=-1)
            _, indices  = torch.topk(output, 5)
            indices = indices[:, -1, 0]
            indices = indices.reshape(1, indices.shape[0])
            translated_sentence = torch.cat((translated_sentence, indices), dim=0)
            self.max_length +=1
        translated_sentence = translated_sentence.squeeze(1)
        return translated_sentence
    

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


##load pixels
img = cv2.imread('./hi.jpg')
img = cv2.resize(img, img_size)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img)
img = np.expand_dims(img, 0).astype('float32')
img = torch.Tensor(img).to(device)
img = torch.permute(img, (0, 3, 1, 2))

dir_img = "./hi.jpg"
convert_tensor = transforms.ToTensor()

image = Image.open(dir_img).convert("RGB")
image = image.resize(img_size)
pixel_values = convert_tensor(image)
pixel_values = pixel_values.unsqueeze(0)
pixel_values = pixel_values.to(device)


end2end_model = EncoderDecoderModel(decoder=decoder, 
                                    fc=fc, 
                                    pos_enc=pos_enc, 
                                    embed_tgt=embed_tgt, 
                                    encoder=encoder, 
                                    enc_to_dec_proj=enc_to_dec_proj)

end2end_model.to(device)
end2end_model.eval()

output = end2end_model(pixel_values)
print(output)
# try:
#     # torch.onnx.export(decoder_merge,               # model being run
#     #                     args=(tgt_inp, encoder_hidden_states),                         # model input (or a tuple for multiple inputs)
#     #                     f = "decoder_merge.onnx",   # where to save the model (can be a file or file-like object)
#     #                     export_params=True,        # store the trained parameter weights inside the model file
#     #                     opset_version=14,          # the ONNX version to export the model to
#     #                     do_constant_folding=True,  # whether to execute constant folding for optimization
#     #                     input_names = ['tgt_inp', 'encoder_hidden_states'],   # the model's input names
#     #                     output_names = ['output_decoder'], # the model's output names
#     #                     dynamic_axes={'tgt_inp' : {0: 'num_loop', 1 : 'batch_size'},    # variable length axes
#     #                                 'encoder_hidden_states' : {1 : 'batch_size'},    # variable length axes
#     #                                 'output_decoder' : {0 : 'batch_size'}})
#     torch.onnx.export(end2end_model,               # model being run
#                     args=(pixel_values),                         # model input (or a tuple for multiple inputs)
#                     f = "./models/encoder_decoder_256.onnx",   # where to save the model (can be a file or file-like object)
#                     export_params=True,        # store the trained parameter weights inside the model file
#                     opset_version=14,          # the ONNX version to export the model to
#                     do_constant_folding=True,  # whether to execute constant folding for optimization
#                     input_names = ['input'],   # the model's input names
#                     output_names = ['output'], # the model's output names
#                     dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
#                                 'output' : {0 : 'batch_size'}})

# except Exception as e:
#     raise e
