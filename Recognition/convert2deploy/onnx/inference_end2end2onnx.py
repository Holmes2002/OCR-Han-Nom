import onnx 
import onnxruntime
import numpy as np
import torch
from torch.nn.functional import log_softmax, softmax
from union.load_config import Cfg
from union.utils import Vocab
from PIL import Image
from torchvision import transforms


config = Cfg.load_config_from_file("./univietocr.yml")
vocab = config.vocab
device = torch.device(config.device)
img_size = (config.width_size, config.height_size)



model = onnxruntime.InferenceSession('./models/encoder_decoder_256.onnx' , providers = ['CPUExecutionProvider'])

dir_img = "./hi.jpg"
convert_tensor = transforms.ToTensor()

image = Image.open(dir_img).convert("RGB")
image = image.resize(img_size)
pixel_values = convert_tensor(image)
pixel_values = pixel_values.unsqueeze(0)
pixel_values = pixel_values.to(device)

pixel_values = np.array(pixel_values)

feed_dict = {model.get_inputs()[0].name: pixel_values}
s = model.run(None, feed_dict)[0]



vocab = Vocab(chars=config.vocab, max_target_length= config.max_length_token)
s = np.asarray(s).T
s = s.tolist()
print(s)
output_text = vocab.batch_decode(s)
print(output_text)

# from icocr import icocr
# import cv2
# import time
# from fastapi import FastAPI, File, UploadFile
# import numpy as np

# app = FastAPI(
#     title="API TEST GET FRAME",
#     description="API TEST GET FRAME"
# )


# @app.post("/trocr/", tags=["trocr"])
# async def trocr(file: UploadFile = File(...)):
#     image_content = await file.read()
#     image_nparray = np.fromstring(image_content, np.uint8)
#     s_time = time.time()
#     image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
#     t1 =time.time()

#     feed_dict = {model.get_inputs()[0].name: image}
# 	s = model.run(None, feed_dict)[0]
    
#     print(time.time() -t1)
#     return out