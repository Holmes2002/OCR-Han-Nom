import cv2
import onnxruntime
import argparse
import icocr.craft_utils as craft_utils
import icocr.imgproc as imgproc
import numpy as np


class TextDet():
    def __init__(self, craft_onnx_path: str, refine_onnx_path: str, use_cuda: False):
        self.load_model_onnx(craft_onnx_path, refine_onnx_path, use_cuda)

    def load_model_onnx(self, craft_onnx_path, refine_onnx_path, use_cuda): 
        if use_cuda:
            self.sess_craft = onnxruntime.InferenceSession(craft_onnx_path, providers = ['CUDAExecutionProvider'])
            self.sess_refinet = onnxruntime.InferenceSession(refine_onnx_path, providers = ['CUDAExecutionProvider'])
        else:
            self.sess_craft = onnxruntime.InferenceSession(craft_onnx_path, providers = ['CPUExecutionProvider'])
            self.sess_refinet = onnxruntime.InferenceSession(refine_onnx_path, providers = ['CPUExecutionProvider'])

        self.craft_input_name = self.sess_craft.get_inputs()[0].name
        self.refi_nput_name_y = self.sess_refinet.get_inputs()[0].name
        self.refi_input_name_feature = self.sess_refinet.get_inputs()[1].name 

    def inference(self, img: np.ndarray):
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(img, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
        ratio_h = ratio_w = 1 / target_ratio
        x = imgproc.normalizeMeanVariance(img_resized)
        # x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        # x = x.unsqueeze(0)                # [c, h, w] to [b, c, h, w]
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)

        y, feature = self.sess_craft.run(None, {self.craft_input_name: x})
        # score_link = torch.tensor(self.sess_refinet.run(None, {self.refi_nput_name_y: y, self.refi_input_name_feature: feature})[0])
        # score_link = score_link[0,:,:,0].cpu().data.numpy()
        score_link = self.sess_refinet.run(None, {self.refi_nput_name_y: y, self.refi_input_name_feature: feature})[0]
        score_link = score_link[0, :, :, 0]
        score_link = np.array(score_link)
        
        ratio_w = ratio_h
        score_text = y[0,:,:,0]
        
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, 0.5, 0.4, 0.4, False)
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        return boxes, polys

