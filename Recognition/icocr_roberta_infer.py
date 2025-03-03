from union.mix_union_vietocr import UnionRoBerta
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
from tqdm import tqdm
import os
import glob
import cv2
import random
def inference(image, img_size = (128,32), device = 'cuda', vocab = '', model = None):
    image = Image.fromarray(image)
    image = image.rotate(90, expand=True)
    image = image.resize(img_size)
    ####
    # os.makedirs('text_recog_Cyber', exist_ok = True)
    # index = len(os.listdir('text_recog_Cyber'))
    # image.save(f'text_recog_Cyber/{index}.jpg')
    ####

    convert_tensor = transforms.ToTensor()
    
    pixel_values = convert_tensor(image)
    pixel_values = pixel_values.unsqueeze(0)
    pixel_values = pixel_values.to(device)
    generated_text = model.generate(pixel_values)
    return generated_text.strip()

def test_train(model, config):
    
    # image = cv2.imread('/home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/dataset/detection/Patches/Pseudo_Labels/BNTwEHieafbaJ1879.1.11_obj_1.jpg')
    # HanNom_vocab = open('/home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/dataset/detection/Patches/vocab.txt', 'r').read().splitlines()
    # text = inference_end2end([image])
    # print(text)
    convert_tensor = transforms.ToTensor()
    root = '/home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/dataset/detection/Patches'
    file_val = open('/home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/dataset/detection/Patches/Val_real.txt', 'r').read().splitlines()
    img_size = (config.width_size, config.height_size)
    random.shuffle(file_val)
    for line in tqdm(file_val):
        name_image, label = line.split('\t')
        image = cv2.imread('/home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/dataset/detection/Patches/Pseudo_Labels/BNTwEHieafbaJ1879.1.11_obj_17.jpg')[:,:,::-1]
        text = inference(image,img_size = img_size, vocab = '', model = model)
        break
    return text, label