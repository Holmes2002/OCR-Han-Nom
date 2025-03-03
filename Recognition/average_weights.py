from union.mix_union_vietocr import UnionVietOCR, UnionRoBerta, TrRoBerta, TrRoBerta_custom
from PIL import Image
import torch
import numpy as np 
from trvietocr.utils import Vocab
from torch.nn.functional import log_softmax, softmax
from torchvision import transforms
import time
import Levenshtein
from trvietocr.load_config_trvietocr import Cfg
import os
from transformers import AutoTokenizer, AutoModel
config = Cfg.load_config_from_file("/home1/vudinh/NomNaOCR/icocr/config/trrobertaocr_512x48.yml")
vocab = config.vocab
device = config.device
HanNom_vocab = open('/home1/vudinh/NomNaOCR/HanNom_Vocab/unique_vocab_and_opensource.txt').read().splitlines()
# HanNom_vocab = open('vocab_tmp.txt').read().splitlines()
tokenizer = AutoTokenizer.from_pretrained("KoichiYasuoka/roberta-small-japanese-aozora-char")
tokenizer = tokenizer.train_new_from_iterator(HanNom_vocab, vocab_size=len(HanNom_vocab))
model = TrRoBerta_custom(max_length_token=config.max_length_token,
                        img_width=config.width_size,
                        img_height=config.height_size,
                        patch_size=config.patch_size,
                        tokenizer = tokenizer,
                        type = 'v2',
                        embed_dim_vit=config.embed_dim_vit
                        )
all_checkpoints = [f'{i}' for i in os.listdir("/home1/vudinh/NomNaOCR/icocr/trocr/pretrain_512x48") if 'pth' in i]
lst_ckpt = sorted(all_checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))[-5:]
lst_ckpt = [f'/home1/vudinh/NomNaOCR/icocr/trocr/pretrain_512x48/{i}' for i in lst_ckpt]
model_dir = lst_ckpt[0]
print(model_dir)
model.load_state_dict(torch.load(model_dir))
std = model.state_dict()

for i in range(1, len(lst_ckpt)):
    model_dir = lst_ckpt[i]
    model.load_state_dict(torch.load(model_dir))
    std_cache = model.state_dict()
    for key in std_cache:
        std[key] = std[key] + std_cache[key]
    
for key in std:
    std[key] = std[key] /(len(lst_ckpt))

torch.save(std, "average_epoch.pth")