import onnxruntime
import numpy as np
from PIL import Image
from icocr.utils import Vocab

class TextRecog():
    def __init__(self, encoder_path: str, decoder_path: str, vocab: str, max_length_token: int,  use_cuda: False):
        if use_cuda:
            self.encoder_model = onnxruntime.InferenceSession(encoder_path, providers = ['CUDAExecutionProvider'])
            self.decoder_model = onnxruntime.InferenceSession(decoder_path, providers = ['CUDAExecutionProvider'])
        else:
            self.encoder_model = onnxruntime.InferenceSession(encoder_path, providers = ['CPUExecutionProvider'])
            self.decoder_model = onnxruntime.InferenceSession(decoder_path, providers = ['CPUExecutionProvider'])

        self.vocab = Vocab(vocab, max_target_length= max_length_token)

    def inference(self, pixel_values, max_seq_length=256, sos_token=1, eos_token=2):
        num_batch = pixel_values.shape[0]
        feed_dict_encoder = {self.encoder_model.get_inputs()[0].name: pixel_values}
        encoder_hidden_states = self.encoder_model.run(None, feed_dict_encoder)
        translated_sentence = [[sos_token]*num_batch]
        max_length = 1
        
        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):
            # tgt_inp = torch.LongTensor(translated_sentence).to(device)
            # tgt_inp = np.array(tgt_inp)
            tgt_inp = np.array(translated_sentence)
            feed_dict_decoder = {self.decoder_model.get_inputs()[0].name: tgt_inp, self.decoder_model.get_inputs()[1].name: encoder_hidden_states[0]}
        
            indices = np.array(self.decoder_model.run(None, feed_dict_decoder))[0]

            indices = indices.tolist()

            translated_sentence.append(indices)   

            max_length += 1
        translated_sentence = np.asarray(translated_sentence).T
        translated_sentence = translated_sentence.tolist()
        output_text = self.vocab.batch_decode(translated_sentence)
        return output_text
