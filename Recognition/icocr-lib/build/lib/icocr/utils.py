import os
import yaml
import numpy as np
import cv2


class Cfg(dict):
    def __init__(self, config_dict):
        super(Cfg, self).__init__(**config_dict)
        self.__dict__ = self

    @staticmethod
    def load_config_from_file(fname):
        base_config = {}
        with open(fname, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        base_config.update(config)

        return Cfg(base_config)

class Vocab():
    def __init__(self, chars, max_target_length):
        self.pad = 0
        self.go = 1
        self.eos = 2
        self.mask_token = 3

        self.chars = chars

        self.c2i = {c:i+4 for i, c in enumerate(chars)}

        self.i2c = {i+4:c for i, c in enumerate(chars)}
        
        self.i2c[0] = '<pad>'
        self.i2c[1] = '<sos>'
        self.i2c[2] = '<eos>'
        self.i2c[3] = '*'
        self.max_target_length = max_target_length
    
    def encode(self, chars):
        return [self.go] + [self.c2i[c] for c in chars] + [self.eos]

    def decode(self, ids):
        first = 1 if self.go in ids else 0
        last = ids.index(self.eos) if self.eos in ids else None
        sent = ''.join([self.i2c[i] for i in ids[first:last]])
        return sent
    
    def __len__(self):
        return len(self.c2i) + 4
    
    def batch_decode(self, arr):
        texts = [self.decode(ids) for ids in arr]
        return texts

    def __str__(self):
        return self.chars


def crop_rect(img, rect, degree, size):
    center = rect[0]
    # get the parameter of the small rectangle
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D(center, degree, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))
    out = cv2.getRectSubPix(img_rot, size, center)
    return out, img_rot

def preprocess_recog(lst_image, width_size, height_size):
    total_img = len(lst_image)
    batch = np.zeros((total_img, 3, height_size, width_size), dtype = np.float32)
    for i in range(total_img):
        image = lst_image[i].resize((width_size, height_size))
        image = np.array(image)
        image = image.astype('float32') / 255.0  # 0 - 255 to 0.0 - 1.0
        image = image.transpose(2, 0, 1)
        batch[i] = image
    return batch

