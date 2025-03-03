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

def univietocr_trans(pixel_values, model, max_seq_length=128, sos_token=1, eos_token=2, return_score = False):
    # model.to('cuda')
    model.eval()
    device = pixel_values.device
    start = time.time()
    scores = []
    with torch.no_grad():
        encoder_outputs = model.vit_model(pixel_values)
        encoder_hidden_states = model.enc_to_dec_proj(encoder_outputs)
        encoder_hidden_states = encoder_hidden_states.transpose(1,0)

        # print(encoder_hidden_states.size())
        # assert False
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
            scores.append(values)


            translated_sentence.append(indices)   

            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T
        scores = np.asarray(scores).T
    end = time.time()
    number_tokenn = len(translated_sentence[0])
    if return_score:
        return translated_sentence, scores
    return translated_sentence


from tqdm import tqdm
import os


import glob

def calculate_accuracy(pred, label):
    correct_chars = sum([1 for p, l in zip(pred, label) if p == l])
    accuracy = correct_chars / len(label) if len(label) > 0 else 0
    return accuracy*100
def calculate_cer(pred, label):
    cer = Levenshtein.distance(pred, label) / len(label) if len(label) > 0 else 0
    return cer

def eval(val_dataloadder, model, config, vocab, device = 'cuda'):
    num_image = len(val_dataloadder)
    avg_acc = 0
    avg_cer = 0
    avg_score = 0
    vocab = Vocab(chars=vocab, max_target_length= config.max_length_token)
    for i, batch in enumerate(tqdm(val_dataloadder)):
        for k,v in batch.items():
            batch[k] = v.to(device)
        pixel_values = batch['img']
        s,scores = univietocr_trans(pixel_values, model, return_score = True)
        s = s[0].tolist()
        output_text = vocab.decode(s).strip()
        list_index_label = [int(i[0]) for i in batch['tgt_input']]
        label = vocab.decode(list_index_label).strip()
        avg_acc += calculate_accuracy(output_text, label)
        avg_cer += calculate_cer(output_text, label)
        avg_score += 0
    avg_cer /= num_image
    avg_acc /= num_image
    avg_score /= num_image
    return avg_acc, avg_cer, avg_score
      
def inference(image, img_size = (128,32), device = 'cuda', vocab = '', model = None):
    image = Image.fromarray(image)
    image = image.rotate(90, expand=True)
    image = image.resize(img_size)
    convert_tensor = transforms.ToTensor()
    pixel_values = convert_tensor(image)
    pixel_values = pixel_values.unsqueeze(0)
    pixel_values = pixel_values.to(device)
    s, scores = univietocr_trans(pixel_values, model, return_score = True)
    vocab = Vocab(vocab, max_target_length= 128)
    s = s[0].tolist()
    scores = scores[0].tolist()[:-1]
    output_text = vocab.decode(s)
    return output_text,sum(scores) /len(scores)
def inference_end2end(list_images, type = 'v1'):
    if type == 'v1':
        HanNom_vocab = open('/home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/dataset/detection/Patches/CLC_and_Synthesis_vocab.txt', 'r').read().splitlines()
    else:
        HanNom_vocab = open('/home1/vudinh/NomNaOCR/HanNom_Vocab/HanNom_vocab.txt', 'r').read().splitlines()
    if type == 'v1':
        config = Cfg.load_config_from_file("/home1/vudinh/NomNaOCR/icocr/config/univietocr.yml")
        model = UnionVietOCR(max_length_token=config.max_length_token,
                            img_width=config.width_size,
                            img_height=config.height_size,
                            patch_size=config.patch_size,
                            vocab_leng = len(HanNom_vocab),
                            type = type

                            )
        model.load_state_dict(torch.load("/home1/vudinh/NomNaOCR/icocr/union/ckpt/256x32.pth"))
    else:
        config = Cfg.load_config_from_file("/home1/vudinh/NomNaOCR/icocr/config/univietocr_HanNom.yml")
        model = UnionVietOCR(max_length_token=config.max_length_token,
                            img_width=config.width_size,
                            img_height=config.height_size,
                            patch_size=config.patch_size,
                            vocab_leng = len(HanNom_vocab),
                            type = type

                            )
        model.load_state_dict(torch.load("/home1/vudinh/NomNaOCR/icocr/union/ckpt_2/epoch_3.pth"))
        # model.load_state_dict(torch.load("/home1/vudinh/NomNaOCR/icocr/union/ckpt_2/256x48_2.pth"))

    model.to('cuda')
    list_texts = []
    list_scores = []
    for image in list_images:
        text, score = inference(image,img_size = (config.width_size, config.height_size), vocab = HanNom_vocab, model = model)
        list_texts.append(text)
        list_scores.append(score)
        # print(text)
        # cv2.imwrite('sample.jpg', image)
        # assert False
    return list_texts, list_scores

def inferece_IC_Roberta(cropped_images):
    config = Cfg.load_config_from_file("/home1/vudinh/NomNaOCR/icocr/config/uniroberta_2.yml")
    vocab = config.vocab
    device = config.device
    HanNom_vocab = open('/home1/vudinh/NomNaOCR/HanNom_Vocab/unique_vocab.txt').read().splitlines()
    # HanNom_vocab = open('vocab_tmp.txt').read().splitlines()
    tokenizer = AutoTokenizer.from_pretrained("KoichiYasuoka/roberta-small-japanese-aozora-char")
    tokenizer = tokenizer.train_new_from_iterator(HanNom_vocab, vocab_size=len(HanNom_vocab))
    model = UnionRoBerta(max_length_token=config.max_length_token,
                        img_width=config.width_size,
                        img_height=config.height_size,
                        patch_size=config.patch_size,
                        tokenizer = tokenizer,
                        type = 'v2',
                        embed_dim_vit=config.embed_dim_vit
                        )
    model.load_state_dict(torch.load('/home1/vudinh/NomNaOCR/icocr/union_roberta/ckpt_320x48_finetune/epoch_1_23999.pth'), strict = False)
    model.to(device)
    tensor_images = []
    for image in cropped_images:
        # Convert numpy array to PIL Image
        image = Image.fromarray(image)
        # Rotate, resize, and convert to RGB
        image = image.rotate(90, expand=True)
        image = image.resize((320,48))  # Resize to (width, height)
        image = image.convert("RGB")  # Ensure 3 channels
        # Convert to tensor and normalize
        image_tensor = transforms.ToTensor()(image)  # Converts to [0, 1] normalized tensor
        tensor_images.append(image_tensor.unsqueeze(0))  # Add batch dimension

    # Combine into a single tensor
    tensor_images = torch.cat(tensor_images, dim=0).to(device)  # Shape: (batch_size, channels, height, width)
    decoded_texts = model.inference(tensor_images, inference = True)
    return decoded_texts
def inferece_TrOCR_Roberta(cropped_images, checkpoint_path = '/home1/vudinh/NomNaOCR/weights/recognition/TrRoberta_custom_512x48.pth'):
    config = Cfg.load_config_from_file("/home1/vudinh/NomNaOCR/icocr/config/trrobertaocr_512x48.yml")
    vocab = config.vocab
    device = config.device
    HanNom_vocab = open('/home1/vudinh/NomNaOCR/HanNom_Vocab/unique_vocab_and_opensource.txt').read().splitlines()
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
    model.load_state_dict(torch.load(checkpoint_path), strict = False)
    model.to(device)
    tensor_images = []
    list_decode_texts = []
    for image in cropped_images:
        try:
            # Convert numpy array to PIL Image
            image = Image.fromarray(image)
            # Rotate, resize, and convert to RGB
            image = image.rotate(90, expand=True)
            image = image.resize((512,48))  # Resize to (width, height)
            image = image.convert("RGB")  # Ensure 3 channels
            # Convert to tensor and normalize
            image_tensor = transforms.ToTensor()(image)  # Converts to [0, 1] normalized tensor
            tensor_images = (image_tensor.unsqueeze(0)).to(device)  # Add batch dimension
            decoded_texts = model.inference(tensor_images, inference = True)
            list_decode_texts.append(decoded_texts[0])
        except : 
            list_decode_texts.append('')
            continue
    return list_decode_texts

if __name__ == "__main__":
    import cv2
    image = cv2.imread('/home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/dataset/detection/Patches/DVSKTT-5 Ban ky tuc bien/DVSKTT_ban_tuc_XIX_10a_3.jpg')
    inferece_TrOCR_Roberta([image])