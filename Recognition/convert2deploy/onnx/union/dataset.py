import torch 
from PIL import Image
from torch.utils.data import Dataset
import os 
import json
from transformers import TrOCRProcessor
from transformers import AutoModel, AutoTokenizer
# from aug_method import RandomAugData
from trvietocr.custom_aug import RandomAugData
from tqdm import tqdm
from random import random, randrange
import numpy as np
import cv2
from trvietocr.utils import Vocab
from torchvision import transforms

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
        if dir_imgs[:4] in "./data":
            dir_imgs = "/home/data2/thaitran/Research/OCR/source/Implement_TrOCR" + dir_imgs[1:]
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

class ICOCRDataset(Dataset):
    def __init__(self, data_dir, chars: str, data_type: str, processor: TrOCRProcessor, max_target_length=128, load_file: bool=True):
        """
        Dataset OCR Custom
        arg:
            :- data_dir         : list path images or path file data
            :- processor        : processor transformes
            :- max_targer_length: maxl length token
            :- data_type        : branch dataloader
        """
        self.processor = processor
        self.max_target_length = max_target_length
        self.vocab = processor.tokenizer.get_vocab()
        self.data_type = data_type
        self.aug_method = RandomAugData(lst_aug_types = ['warp', 'geometry', 'blur', 'noise', 'camera'],
                                        prob_list = [0.1, 0.1, 0.45, 0.0, 0.3])
        assert self.data_type in ["train", "val", "test"], "Data type is 'train' or 'val'"
        self.labels = load_fileimages_and_labels(data_dir, load_file)
        self.lst_imgs_path = list(self.labels.keys())
        self.lst_labels = list(self.labels.values())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = Vocab(chars=chars, max_target_length= self.max_target_length)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # get images and text labels
        img_path = self.lst_imgs_path[idx]
        text = self.lst_labels[idx]

        image = Image.open(img_path).convert("RGB")
        
        # if self.data_type == "train":
        #     image = self.aug_method(image)
        # pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # labels = encode_text(text, max_target_length=self.max_target_length, vocab=self.vocab)
        image = image.resize((128, 32))
        img = self.convert_tensor(image)
        labels = self.vocab.encode(text)
        # labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        encoding = {"pixel_values": img, "labels": torch.tensor(labels)}
        return encoding


class UniVietOCRDataset(Dataset):
    def __init__(self, data_dir, chars: str, data_type: str, img_size: tuple=(128,32), max_target_length=128,load_file: bool=True):
        """
        Dataset OCR Custom
        arg:
            :- data_dir         : list path images or path file data
            :- processor        : processor transformes
            :- max_targer_length: maxl length token
            :- data_type        : branch dataloader
        """
        self.max_target_length = max_target_length
        self.data_type = data_type
        self.aug_method = RandomAugData(lst_aug_types = ['warp', 'geometry', 'blur', 'noise', 'camera'],
                                        prob_list = [0.1, 0.1, 0.45, 0.0, 0.3])
        assert self.data_type in ["train", "val", "test"], "Data type is 'train' or 'val'"
        self.labels = load_fileimages_and_labels(data_dir, load_file)
        self.lst_imgs_path = list(self.labels.keys())
        self.lst_labels = list(self.labels.values())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = Vocab(chars=chars, max_target_length= self.max_target_length)
        self.convert_tensor = transforms.ToTensor()
        self.img_size = img_size
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # get images and text labels
        img_path = self.lst_imgs_path[idx]
        text = self.lst_labels[idx]

        image = Image.open(img_path).convert("RGB")
        # pixel_values = self.processor(image, return_tensors="pt").pixel_values
        image = image.resize(self.img_size)
        img = self.convert_tensor(image)

        labels = self.vocab.encode(text)

        encoding = {"img": img, "word": labels}
        return encoding


class Collator(object):
    def __init__(self, masked_language_model=True):
        self.masked_language_model = masked_language_model

    def __call__(self, batch):
        img = []
        target_weights = []
        tgt_input = []
        max_label_len = max(len(sample['word']) for sample in batch)
        for sample in batch:
            sample["img"] = np.array(sample["img"])
            img.append(sample['img'])
            label = sample['word']
            label_len = len(label)
            
            
            tgt = np.concatenate((
                label,
                np.zeros(max_label_len - label_len, dtype=np.int32)))
            tgt_input.append(tgt)

            one_mask_len = label_len - 1

            target_weights.append(np.concatenate((
                np.ones(one_mask_len, dtype=np.float32),
                np.zeros(max_label_len - one_mask_len,dtype=np.float32))))
            
        img = np.array(img, dtype=np.float32)


        tgt_input = np.array(tgt_input, dtype=np.int64).T
        tgt_output = np.roll(tgt_input, -1, 0).T
        tgt_output[:, -1]=0
        
        # random mask token
        if self.masked_language_model:
            mask = np.random.random(size=tgt_input.shape) < 0.05
            mask = mask & (tgt_input != 0) & (tgt_input != 1) & (tgt_input != 2)
            tgt_input[mask] = 3

        tgt_padding_mask = np.array(target_weights)==0

        rs = {
            'img': torch.FloatTensor(img),
            'tgt_input': torch.LongTensor(tgt_input),
            'tgt_output': torch.LongTensor(tgt_output),
            'tgt_padding_mask': torch.BoolTensor(tgt_padding_mask)
        }   
        
        return rs
