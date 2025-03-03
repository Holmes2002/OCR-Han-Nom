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
from Levenshtein import distance as levenshtein_distance

root_images = '/home1/vudinh/NomNaOCR/icocr/HanNom_dataset_test/text_recognition_test_data/cropped_images'
labels_file = open('/home1/vudinh/NomNaOCR/icocr/HanNom_dataset_test/text_recognition_test_data/labels.txt').read().splitlines()
def eval_score(list_image_paths, list_labels):
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
    model.load_state_dict(torch.load('/home1/vudinh/NomNaOCR/icocr/average_epoch.pth'), strict = False)
    model.to(device)
    
    list_decode_texts = []
    batch_size = 16
    batches = [list_image_paths[i:i + batch_size] for i in range(0, len(list_image_paths), batch_size)]
    for batch in batches:
        tensor_images = []
        for image in batch:
            # Convert numpy array to PIL Image
            image = Image.open(image).convert('RGB')
            # Rotate, resize, and convert to RGB
            image = image.rotate(90, expand=True)
            image = image.resize((512,48))  # Resize to (width, height)
            image = image.convert("RGB")  # Ensure 3 channels
            # Convert to tensor and normalize
            image_tensor = transforms.ToTensor()(image)  # Converts to [0, 1] normalized tensor
            image_tensor = image_tensor.unsqueeze(0).to(device) 
            tensor_images.append(image_tensor)
        tensor_images = torch.cat(tensor_images, dim=0).to(device)
        decoded_texts = model.inference(tensor_images, inference = True)
        list_decode_texts += decoded_texts
        print(decoded_texts)
        print(list_labels[:batch_size])
        print(batch)
        assert False
    ### Write code to calculate Editdistance list_decode_texts and list_labels
    # Calculate Edit Distance
    total_distance = 0
    total_chars = 0
    edit_distances = []

    for predicted, ground_truth in zip(list_decode_texts, list_labels):
        distance = levenshtein_distance(predicted, ground_truth)
        total_distance += distance
        total_chars += len(ground_truth)
        edit_distances.append(distance)

    # Metrics
    average_edit_distance = total_distance / len(list_labels)
    character_error_rate = total_distance / total_chars

    print(f"Average Edit Distance: {average_edit_distance}")
    print(f"Character Error Rate (CER): {character_error_rate}")

    # Optionally, return metrics
    return {
        "average_edit_distance": average_edit_distance,
        "character_error_rate": character_error_rate,
        "edit_distances": edit_distances
    }

if __name__ == '__main__':
    image_names = []
    text_labels = []
    for line in labels_file:
        name, text = line.split('\t')
        image_names.append(f'{root_images}/{name}')
        text_labels.append(text)
    eval_score(image_names, text_labels)
    