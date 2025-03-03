from union.mix_union_vietocr import UnionVietOCR
from trvietocr.load_config_trvietocr import Cfg
from PIL import Image
import torch
import numpy as np 
from trvietocr.utils import Vocab
from torch.nn.functional import log_softmax, softmax
from torchvision import transforms

def univietocr_trans(pixel_values, model, max_seq_length=256, sos_token=1, eos_token=2):
    model.eval()
    device = pixel_values.device
    with torch.no_grad():
        encoder_outputs = model.vit_model(pixel_values)
        encoder_hidden_states = model.enc_to_dec_proj(encoder_outputs)
        encoder_hidden_states = encoder_hidden_states.transpose(1,0)

        translated_sentence = [[sos_token]*1]

        max_length = 1

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):
        # while max_length <= max_seq_length:
            tgt_inp = torch.LongTensor(translated_sentence).to(device)

            output, encoder_hidden_states = model.forward_decoder(tgt_inp, encoder_hidden_states)

            output = softmax(output, dim=-1)
            output = output.to('cpu')

            values, indices  = torch.topk(output, 5)

            indices = indices[:, -1, 0]
            indices = indices.tolist()
            
            values = values[:, -1, 0]
            values = values.tolist()


            translated_sentence.append(indices)   

            max_length += 1

            del output
        translated_sentence = np.asarray(translated_sentence).T

    return translated_sentence

config = Cfg.load_config_from_file("./config/univietocr.yml")
vocab = config.vocab
device = config.device
img_size = (config.width_size, config.height_size)

model = UnionVietOCR(max_length_token=config.max_length_token,
                     img_width=config.width_size,
                     img_height=config.height_size,
                     patch_size=config.patch_size)
model.load_state_dict(torch.load("./union/ckpt/best_decoder.pth"))
model.to(device)
model.eval()


from tqdm import tqdm
import os

def load_fileimages_and_labels(data_dir, load_file: bool=True) -> dict:
    if load_file:
        file = open(data_dir, "r")
        lst_data = file.readlines()
    else:
        lst_data = data_dir
    labels = dict()
    count_domain = dict()
    count = 0
    for data in tqdm(lst_data):
        data = data.split(",", 1)
        dir_imgs = data[0]
        if not os.path.exists(dir_imgs):
            print(dir_imgs)
            count+=1
            continue
        if data[1][-1] == "\n":
            text = data[1][:-1]
        else:
            text = data[1]
        if len(text)>200:
            text = text[0:200]
        labels[dir_imgs] = text
        domain = dir_imgs.split("/")[-2]
        if domain not in count_domain:
            count_domain[domain] = 1
        else:
            count_domain[domain] +=1
    print("Num fail path: ", count)
    print("Num data with Domain: ", count_domain)
    return labels

data = load_fileimages_and_labels("./test_data/test_data_only_printed.txt")
lst_dir_img =list(data.keys())
lst_label = list(data.values())
test_cer = 0
convert_tensor = transforms.ToTensor()

##load vietocr result
data_vietocr = load_fileimages_and_labels("./test_data/vietocr_result.txt")
lst_dir_img_vietocr =list(data_vietocr.keys())
lst_label_vietocr = list(data_vietocr.values())
vietocr_cer = 0
import glob

for i, dir_img in enumerate(lst_dir_img):

    image = Image.open(dir_img).convert("RGB")
    image = image.resize(img_size)
    pixel_values = convert_tensor(image)
    pixel_values = pixel_values.unsqueeze(0)
    pixel_values = pixel_values.to(device)
    s = univietocr_trans(pixel_values, model)
    vocab = Vocab(chars=config.vocab, max_target_length= config.max_length_token)
    s = s[0].tolist()
    output_text = vocab.decode(s)
    label = lst_label[i]
    print("label: ", label)
    print("union: ", output_text)
    vietocr_result = data_vietocr[dir_img]
    print("viocr: ", vietocr_result)
    
    
    cer = cer_metric.compute(predictions=[output_text], references=[label])
    vi_cer = cer_metric.compute(predictions=[vietocr_result], references=[label])
    print(cer)
    print(vi_cer)
    print("======")
    
    test_cer +=cer
    vietocr_cer +=vi_cer

me_cer = str(test_cer/len(lst_dir_img))
vi_me_cer = str(vietocr_cer/len(lst_dir_img))
print("[INFO] Union Cer test dataset: ", me_cer)
print("[INFO] ViOCR Cer test dataset: ", vi_me_cer)



###only uni

# lst_dir_img = glob.glob("./test/*")
# for i, dir_img in enumerate(lst_dir_img):

#     image = Image.open(dir_img).convert("RGB")
#     image = image.resize(img_size)
#     pixel_values = convert_tensor(image)
#     pixel_values = pixel_values.unsqueeze(0)
#     pixel_values = pixel_values.to(device)
#     s = univietocr_trans(pixel_values, model)
#     vocab = Vocab(chars=config.vocab, max_target_length= config.max_length_token)
#     s = s[0].tolist()
#     output_text = vocab.decode(s)
#     label = lst_label[i]
#     print(dir_img)
#     print("label: ", label)
#     print("union: ", output_text)
    
#     cer = cer_metric.compute(predictions=[output_text], references=[label])
#     print(cer)
#     print("======")
    
#     test_cer +=cer

# me_cer = str(test_cer/len(lst_dir_img))
# print("[INFO] Union Cer test dataset: ", me_cer)