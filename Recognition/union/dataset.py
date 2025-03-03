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
from random import random, randrange, randint
import numpy as np
import cv2
from trvietocr.utils import Vocab
from torchvision import transforms

file_vocab = open("/home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/dataset/detection/Patches/vocab_new_2.txt", 'w')
list_vocab = []
def load_fileimages_and_labels(data_dir, load_file: bool=True, data_type: str = 'train') -> dict:
    global list_vocab 
    if load_file:
        file = open(data_dir, "r", encoding = 'utf-8')
        lst_data = file.readlines()
    else:
        lst_data = data_dir
    if data_type != 'train':
        lst_data = lst_data[:500]
    labels = dict()
    count_domain = dict()
    count = 0
    for data in tqdm(lst_data):
        data = data.split("\t", 1)
        # if 'Nom'  in data[0] or '…' in data[1] or 'multi-domain_2' in data[0] or 'multi-domain_3' in data[0] or 'GeneratedImages' in data[0]: continue
        if 'Nom'  in data[0] or '…' in data[1] or 'multi-domain_2' in data[0] or 'multi-domain_3' in data[0] or 'multi-domain_5' in data[0] or 'multi-domain_6' in data[0] or 'GeneratedImages' in data[0]: continue
        if   len(data[1]) <2: continue
        # if 'multi-domain' in data[0]:
        #     if random() < 0.5: continue
        # if 'OpenSource' not in data[0]: continue
        # if 'multi-domain' in data[0]:
        #     if random() < 0.4: continue
        if 'multi-domain_test' in data[0]:
            dir_imgs = f'/home1/vudinh/NomNaOCR/Text-Generator-main/{data[0]}'
        else:
            dir_imgs = f"/home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/dataset/detection/Patches/{data[0]}"
        # if 'OpenSource' in dir_imgs:
        #     continue
        #     dir_imgs = dir_imgs.replace('OpenSource', 'OpenSource_resize')
        # if dir_imgs[:4] in "./data":
        #     dir_imgs = "/home/data2/thaitran/Research/OCR/source/Implement_TrOCR" + dir_imgs[1:]
        if not os.path.exists(dir_imgs):
            count+=1
            continue
        if data[1][-1] == "\n":
            text = data[1][:-1]
        else:
            text = data[1]
        if len(text)>200:
            text = text[0:200]
        if len(text) == 1 and 'CLC' in data[0]: 
            continue
        list_vocab += list(text)
        # if '\\' in text: continue
        # if '\U000f0823' in text or '\U000f000e' in text: assert False
        #     continue
        labels[dir_imgs] = text
        domain = dir_imgs.split("/")[-2]
        if domain not in count_domain:
            count_domain[domain] = 1
        else:
            count_domain[domain] +=1
    list_vocab = list(set(list_vocab ))
    for i in list_vocab: file_vocab.write(i+"\n")
    print("Num fail path: ", count)
    print("Num data with Domain: ", count_domain)
    # assert False
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
        try:
            image = Image.open(img_path).convert("RGB")
            image = image.resize((128, 32))
            img = self.convert_tensor(image)
            labels = self.vocab.encode(text)
        except:
            image = Image.open(img_path).convert("RGB")
            image.save('error.jpg')
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
        self.aug_method = RandomAugData(lst_aug_types = ['blur', 'noise', 'camera'],
                                        prob_list = [ 0.4, 0.3, 0.3])
        assert self.data_type in ["train", "val", "test"], "Data type is 'train' or 'val'"
	
        self.labels = load_fileimages_and_labels(data_dir, load_file, data_type)
        self.lst_imgs_path = list(self.labels.keys())
        self.lst_labels = list(self.labels.values())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = Vocab(chars=chars, max_target_length= self.max_target_length)
        self.convert_tensor = transforms.ToTensor()
        self.img_size = img_size
        self.chars = chars
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # get images and text labels
        img_path = self.lst_imgs_path[idx]
        text = self.lst_labels[idx]

        image = Image.open(img_path).convert("RGB")
        image = image.rotate(90, expand=True)
        image = image.resize(self.img_size)
        if random() < 0.2:
            image = self.aug_method(image)
        # image.save(f'image_{idx}.jpg')
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

class UnionROBERTtaDataset(Dataset):
    def __init__(self, data_dir, chars: str, data_type: str, img_size: tuple=(128,32), max_target_length=128,load_file: bool=True, is_finetune_model = False):
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
        self.aug_method = RandomAugData(lst_aug_types = ['blur', 'noise', 'camera'],
                                        prob_list = [ 0.4, 0.3, 0.3])
        assert self.data_type in ["train", "val", "test"], "Data type is 'train' or 'val'"
	
        self.labels = load_fileimages_and_labels(data_dir, load_file, data_type)
        self.lst_imgs_path = list(self.labels.keys())
        self.lst_labels = list(self.labels.values())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = Vocab(chars=chars, max_target_length= self.max_target_length)
        self.convert_tensor = transforms.ToTensor()
        self.img_size = img_size
        self.chars = chars
        self.is_finetune_model = is_finetune_model
        if self.is_finetune_model:
            self.list_real_dataset = [ [path, text] for path, text in zip(self.lst_imgs_path, self.lst_labels) if 'domain' not in path]
            self.lst_imgs_path_real = [i[0] for i in self.list_real_dataset]
            self.lst_labels_real = [i[1] for i in self.list_real_dataset]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # get images and text labels
        # if not self.is_finetune_model:
        img_path = self.lst_imgs_path[idx]
        text = self.lst_labels[idx]
        # else:
        #     if random() < 0.2:
        #         idx = randint(0, len(self.lst_imgs_path_real) - 1)
        #         img_path = self.lst_imgs_path_real[idx]
        #         text = self.lst_labels_real[idx]
        #         # print(img_path, text)
        #         # assert False
        #     else:
        #         img_path = self.lst_imgs_path[idx]
        #         text = self.lst_labels[idx]

        image = Image.open(img_path).convert("RGB")
        image = image.rotate(90, expand=True)
        image = image.resize(self.img_size)
        if random() < 0.1:
            image = self.aug_method(image)
        # image.save(f'image_{idx}.jpg')
        img = self.convert_tensor(image)
        
        # tgt_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        # encoding = {"img": img, "input_ids": tgt_tokens["input_ids"], 'attention_mask':tgt_tokens["attention_mask"]}
        encoding = {"img": img, "word": text}

        return encoding
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right without including the start token (`/s`).
    The output should exclude the first token and handle padding tokens in the final position.
    
    Args:
        input_ids (torch.Tensor): Input token IDs of shape `(batch_size, seq_len)`.
        pad_token_id (int): Padding token ID.
    
    Returns:
        torch.Tensor: Shifted token IDs.
    """
    # Initialize a new tensor for shifted input IDs
    shifted_input_ids = input_ids.new_full(input_ids.shape, pad_token_id)
    
    # Shift the tokens to the right, excluding the first token (start token)
    shifted_input_ids[:, :-1] = input_ids[:, 1:].clone()
    
    # Ensure padding tokens are properly placed in the final position
    shifted_input_ids[shifted_input_ids == pad_token_id] = pad_token_id

    return shifted_input_ids

class Collator_Roberta(object):
    def __init__(self, custom_vocab):
        self.tokenizer = AutoTokenizer.from_pretrained("KoichiYasuoka/roberta-small-japanese-aozora-char")
        # self.tokenizer = AutoTokenizer.from_pretrained("KoichiYasuoka/roberta-base-japanese-aozora-char")
        
        # custom_vocab = open('/home1/vudinh/NomNaOCR/HanNom_Vocab/unique_vocab.txt').read().splitlines()
        self.tokenizer = self.tokenizer.train_new_from_iterator(custom_vocab, vocab_size=len(custom_vocab))
    def __call__(self, batch):
        img = []
        tgt_input = []
        for sample in batch:
            img.append(np.array(sample["img"]))
            tgt_input.append(sample['word'])
        img = np.array(img, dtype=np.float32)
        tgt_tokens = self.tokenizer(tgt_input, return_tensors="pt", padding=True, truncation=True)
        tgt_output = tgt_tokens["input_ids"].clone()
        tgt_output.masked_fill_(tgt_output == self.tokenizer.pad_token_id, -100)
        # tgt_output = torch.where(
        #     tgt_output == self.tokenizer.pad_token_id,  # Mask padding tokens
        #     torch.tensor(-100),  # Replace padding tokens with -100
        #     tgt_output  # Keep other tokens as-is
        # )
        
        # Optionally mask tokens in input_ids
        # encoding = {"img": torch.FloatTensor(img), "tgt_input": torch.LongTensor(tgt_input), 'tgt_padding_mask':tgt_tokens["attention_mask"], 
        #             'tgt_output': torch.LongTensor(tgt_tokens["input_ids"])} 
        encoding = {"img": torch.FloatTensor(img), "tgt_input": torch.LongTensor(tgt_tokens["input_ids"]), 'tgt_padding_mask':tgt_tokens["attention_mask"], 
                    'tgt_output': torch.LongTensor(tgt_output)} 
        # if 1 in tgt_tokens["input_ids"]: assert False
        # print('tgt_input :',torch.LongTensor(tgt_tokens["input_ids"]))
        # print('tgt_output :',tgt_output)

        # assert False
        # print(tgt_tokens["input_ids"], sample['word'])
        # assert False
        return encoding
