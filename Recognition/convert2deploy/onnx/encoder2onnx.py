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


config = Cfg.load_config_from_file("./univietocr.yml")
vocab = config.vocab
device = config.device
img_size = (config.width_size, config.height_size)
model = UnionVietOCR(max_length_token=config.max_length_token,
                     img_width=config.width_size,
                     img_height=config.height_size,
                     patch_size=config.patch_size,
                     vietocr_pretrained=config.vietocr_pretrained,
                     decoder_vietocr_pretrained=config.decoder_vietocr_pretrained,
                     fc_vietocr_pretrained=config.fc_vietocr_pretrained,
                     union_pretrained=config.union_pretrained)
dir_ckpt = "./average_epoch_22.pth"
model.load_state_dict(torch.load(dir_ckpt))
model.to(device)
model.eval()

encoder = model.vit_model
enc_to_dec_proj = model.enc_to_dec_proj
decoder = model.decoder
fc = model.fc

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
        print(encoder_hidden_states.shape)
        return encoder_hidden_states
    
encoder_merge = EncoderModel(encoder=encoder, enc_to_dec_proj=enc_to_dec_proj)
encoder_merge.eval()


##load pixels
img = cv2.imread('./test.jpg')
img = cv2.resize(img, img_size)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img)
img = np.expand_dims(img, 0).astype('float32')
img = torch.Tensor(img).cuda()
img = torch.permute(img, (0, 3, 1, 2))


dir_img = "./test.jpg"
convert_tensor = transforms.ToTensor()

image = Image.open(dir_img).convert("RGB")
image = image.resize(img_size)
pixel_values = convert_tensor(image)
pixel_values = pixel_values.unsqueeze(0)
pixel_values = pixel_values.to(device)

batch = torch.zeros((1, 3, config.height_size, config.width_size), dtype = torch.float32)
batch[0] = pixel_values



torch.onnx.export(encoder_merge,               # model being run
                    batch,                         # model input (or a tuple for multiple inputs)
                    "./models/encoder_e22.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output_encoder'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output_encoder' : {1 : 'batch_size'}})

