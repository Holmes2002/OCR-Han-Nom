import onnx 
import onnxruntime
import numpy as np
import torch
from torch.nn.functional import log_softmax, softmax
from union.load_config import Cfg
from union.utils import Vocab
from PIL import Image
from torchvision import transforms


encoder_model = onnxruntime.InferenceSession('./encoder_merge.onnx', providers = ['CPUExecutionProvider'])
decoder_model = onnxruntime.InferenceSession('./decoder_merge.onnx', providers = ['CPUExecutionProvider'])

def univietocr_trans(pixel_values, encoder_model, decoder_model, max_seq_length=256, sos_token=1, eos_token=2):
    device = pixel_values.device
    num_batch = pixel_values.shape[0]
    pixel_values = np.array(pixel_values)

    feed_dict_encoder = {encoder_model.get_inputs()[0].name: pixel_values}
    encoder_hidden_states = encoder_model.run(None, feed_dict_encoder)

    translated_sentence = [[sos_token]*num_batch]

    max_length = 1
    
    while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):
        tgt_inp = torch.LongTensor(translated_sentence).to(device)
        tgt_inp = np.array(tgt_inp)
        feed_dict_decoder = {decoder_model.get_inputs()[0].name: tgt_inp, decoder_model.get_inputs()[1].name: encoder_hidden_states[0]}
    
        indices = np.array(decoder_model.run(None, feed_dict_decoder))[0]

        # output = torch.tensor(output)
        # output = softmax(output, dim=-1)

        # output = output.to('cpu')

        # values, indices  = torch.topk(output, 5)

        # indices = indices[:, -1, 0]
        indices = indices.tolist()
        
        # values = values[:, -1, 0]
        # values = values.tolist()
        # print(indices)
        # print(translated_sentence)

        translated_sentence.append(indices)   

        max_length += 1

        # del output
        # print(translated_sentence)
    translated_sentence = np.asarray(translated_sentence).T

    return translated_sentence

config = Cfg.load_config_from_file("./univietocr.yml")
vocab = config.vocab
device = torch.device(config.device)
img_size = (config.width_size, config.height_size)

if __name__ == "__main__":

    dir_img = "./hi.jpg"
    convert_tensor = transforms.ToTensor()

    image = Image.open(dir_img).convert("RGB")
    image = image.resize(img_size)
    pixel_values = convert_tensor(image)
    pixel_values = pixel_values.unsqueeze(0)
    pixel_values = pixel_values.to(device)

    batch = torch.zeros((1, 3, config.height_size, config.width_size), dtype = torch.float32)
    batch[0] = pixel_values
    # batch[1] = pixel_values

    s = univietocr_trans(pixel_values=batch, encoder_model=encoder_model, decoder_model=decoder_model)
    vocab = Vocab(chars=config.vocab, max_target_length= config.max_length_token)
    s = s[0].tolist()
    output_text = vocab.decode(s)
    print(output_text)