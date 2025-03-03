import yaml
from exec_backends.encoder_trt_backend import EncoderTrtModel
from exec_backends.decoder_trt_backend import DecoderTrTModel
import numpy as np

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

class ICOCRTensorRT():
    def __init__(self, vocab: str, encoder_dir: str, decoder_dir: str, max_seq_length: int=128):
        self.encoder_dir = encoder_dir
        self.decoder_dir = decoder_dir
        self.max_seq_length = max_seq_length
        self.load_trt_model()
        self.vocab = Vocab(vocab, max_target_length= self.max_seq_length)

    def load_trt_model(self):
        self.encoder_model = EncoderTrtModel(self.encoder_dir, 128)
        self.decoder_model = DecoderTrTModel(self.decoder_dir)

    def inference(self, img, sos_token=1, eos_token=2):
        out_encoder = self.encoder_model.run(img)[0]
        max_length = 1
        num_batch = img.shape[0]
        translated_sentence = [[sos_token]*num_batch]

        while max_length <= self.max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):
            tgt_inp = np.array(translated_sentence)
            out_decoder = self.decoder_model.run(tgt_inp, out_encoder)
            out_decoder = out_decoder.tolist()
            translated_sentence.append(out_decoder)
            max_length += 1

        translated_sentence = np.asarray(translated_sentence).T
        translated_sentence = translated_sentence.tolist()
        output_text = self.vocab.batch_decode(translated_sentence)
        return output_text

if __name__ == "__main__":
    import cv2
    bgr_img = cv2.imread("/home1/data/thaitran/Research/OCR/source/end2end_univietocr/vietocr/convert2deploy/onnx/hi.jpg")
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (128, 32))
    inp = rgb_img.astype('float32') / 255.0  # 0 - 255 to 0.0 - 1.0

    bgr_img = cv2.imread("/home1/data/thaitran/Research/OCR/source/end2end_univietocr/vietocr/convert2deploy/onnx/hi1.jpg")
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (128, 32))
    inp1 = rgb_img.astype('float32') / 255.0  # 0 - 255 to 0.0 - 1.0

    inp2 = rgb_img.astype('float32') / 255.0  # 0 - 255 to 0.0 - 1.0
    inp3 = rgb_img.astype('float32') / 255.0  # 0 - 255 to 0.0 - 1.0
    
    batch_inp = np.zeros((4, 3, 32, 128), dtype=np.float32)
    batch_inp[0] = inp.transpose(2, 0, 1)
    batch_inp[1] = inp1.transpose(2, 0, 1)
    batch_inp[2] = inp1.transpose(2, 0, 1)
    batch_inp[3] = inp1.transpose(2, 0, 1)

    config = Cfg.load_config_from_file("./config.yml")
    vocab = config.vocab

    encoder_trt = "/home1/data/thaitran/Research/OCR/source/end2end_univietocr/vietocr/convert2deploy/trt/models/encoder_merge.trt"
    decoder_trt = "/home1/data/thaitran/Research/OCR/source/end2end_univietocr/vietocr/convert2deploy/trt/models/modelsdecoder_merge.trt"

    icocr_trt = ICOCRTensorRT(vocab, encoder_trt, decoder_trt)
    import time
    for i in range(0, 1000):
        t1 = time.time()
        print(icocr_trt.inference(batch_inp))
        print(time.time() - t1)




