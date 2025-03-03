import os
import glob
import shutil
from tqdm import tqdm
from transformers import TrOCRProcessor
import unicodedata
from utils import Vocab
from load_config_trvietocr import Cfg

file_data = open("data_17_7_without_unknow.txt", "r")
data = file_data.readlines()
file_data.close()
config = Cfg.load_config_from_file("../config/trvietocr.yml")
# processor = TrOCRProcessor.from_pretrained(config.processor_pretrained_path) 
# vocab = processor.tokenizer.get_vocab()
count_unknown = 0
count_path_fail = 0
print(len(data))
# print(vocab)
import time
# time.sleep(30)
count=0
count_fail_type = 0
count_remove = 0

vocab =  Vocab(chars=config.vocab, max_target_length=256)
for line in tqdm(data):
    lst_info = line.split(",", 1)
    img_dir = lst_info[0]
    label_before   = lst_info[1]
    # if label[:-1]=="\n":
    # 	label = label[:-2]
    label_before = label_before.replace("\n", "")
    label = unicodedata.normalize('NFC', label_before)
    if label != label_before:
        count_fail_type +=1
        continue
    if img_dir[:4] in "./data":
        img_dir = "/home/data2/thaitran/Research/OCR/source/Implement_TrOCR" + img_dir[1:]
    if not os.path.isfile(img_dir): 
        count_path_fail +=1
        print(img_dir)
        continue
    # embedd = encode_text(label, max_target_length=256, vocab=vocab)
    embedd = vocab.encode(label)
    if -1 in embedd:
        count_unknown +=1
        continue
    if len(label)>200:
        count+=1
        label = label[0:200]
    
    if len(label)>124:
        count_remove+=1
        continue



    file = open("data_17_7.txt", "a")
    word = img_dir + "," + label + "\n"
    file.write(word)
    file.close()

print("Unknown:   ", count_unknown)
print("Fail path: ", count_path_fail)
print("Long data: ", count)
print("fail type: ", count_fail_type)
print("long data: ", count_remove)
file = open("data_17_7.txt", "r")
data = file.readlines()
file.close()
print(len(data))


# processor = TrOCRProcessor.from_pretrained(config.pretrained_processor_path) 
# vocab = processor.tokenizer.get_vocab()
# embedd = encode_text("Ülabekl", max_target_length=256, vocab=vocab)
# print(embedd)
# print(3 in embedd)
