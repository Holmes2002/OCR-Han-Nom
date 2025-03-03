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
    _ = model.generate(pixel_values)
    assert False
    return output_text,sum(scores) /len(scores)


if __name__ == "__main__":
    import cv2
    # image = cv2.imread('/home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/dataset/detection/Patches/Pseudo_Labels/BNTwEHieafbaJ1879.1.11_obj_1.jpg')
    # HanNom_vocab = open('/home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/dataset/detection/Patches/vocab.txt', 'r').read().splitlines()
    # text = inference_end2end([image])
    # print(text)
    convert_tensor = transforms.ToTensor()

    # cv2.imwrite('sample.jpg', image)
    root = '/home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/dataset/detection/Patches'
    file_val = open('/home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/dataset/detection/Patches/Val_real.txt', 'r').read().splitlines()
    HanNom_vocab = open('/home1/vudinh/NomNaOCR/HanNom_Vocab/unique_vocab.txt').read().splitlines()

    config = Cfg.load_config_from_file("./config/uniroberta.yml")
    model = UnionRoBerta(max_length_token=config.max_length_token,
                     img_width=config.width_size,
                     img_height=config.height_size,
                     patch_size=config.patch_size,
                     custom_vocab = (HanNom_vocab),
                     type = 'v2',
                     embed_dim_vit=config.embed_dim_vit

                         )
    model.load_state_dict(torch.load("/home1/vudinh/NomNaOCR/icocr/union_roberta/ckpt_320x48/epoch_1.pth"))
    model.to('cuda')
    # file_val = file_val[:5]
    img_size = (config.width_size, config.height_size)
    for line in tqdm(file_val):
        name_image, label = line.split('\t')
        image = cv2.imread(f'{root}/{name_image}')[:,:,::-1]

        text = inference(image,img_size = img_size, vocab = HanNom_vocab, model = model)
        print(text, label)
        assert False
    print(avg_cer/len(file_val))
    print(avg_acc/len(file_val))
