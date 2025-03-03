import numpy as np 
# import pandas as pd 
# import onnxruntime
from text_det.exec_backend.triton_backend import Craftmlt25kTextDet, RefineTextDet
import cv2
from text_det import craft_utils
from text_det import imgproc
import torch
class RefiTextDet(object):
    def __init__(self,
                # model_path='weights/imintv3.trt',
                input_shape = (192, 192),
                batch_size = 1,
                engine = 'TRT',
                triton_model_name = 'refine_text_det',
                triton_protocol = 'GRPC',
                triton_host = "0.0.0.0:8001",
                triton_verbose = False):
        print('[INFO] Create Refine Text model with {} engine'.format(engine))
        self.engine = engine

        self.input_shape = input_shape
        self.batch_size = batch_size

        if self.engine == "TRT":
            if triton_protocol == 'GRPC':
                self.model = RefineTextDet(triton_host = triton_host, # default GRPC port
                                            triton_model_name = triton_model_name,
                                            verbose = triton_verbose)
           
        else:
            raise NotImplementedError("Current support only TRT, ONNX & TRITON engine")

    def infer(self, y, featue):
        y_refiner = self.model.run([y, featue])[0]
        # print(type(y_refiner))
        y_refiner = torch.tensor(y_refiner)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()
        return score_link

class CraftTextDet(object):
    def __init__(self,
                # model_path='weights/imintv3.trt',
                input_shape = (192, 192),
                batch_size = 1,
                engine = 'TRT',
                triton_model_name = 'craft_text_det',
                triton_protocol = 'GRPC',
                triton_host = "0.0.0.0:8001",
                triton_verbose = False):
        print('[INFO] Create Craft text detection model with {} engine'.format(engine))
        self.engine = engine

        self.input_shape = input_shape
        self.batch_size = batch_size

        if self.engine == "TRT":
            if triton_protocol == 'GRPC':
                self.model = Craftmlt25kTextDet(triton_host = triton_host, # default GRPC port
                                            triton_model_name = triton_model_name,
                                            verbose = triton_verbose)
           
        else:
            raise NotImplementedError("Current support only TRT, ONNX & TRITON engine")
    def infer(self, img):
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(img, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
        ratio_h = ratio_w = 1 / target_ratio
        inp = imgproc.normalizeMeanVariance(img_resized)
        # x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        # x = x.unsqueeze(0) 
        inp = inp.transpose(2, 0, 1)  # BGR to RGB
        inp = np.expand_dims(inp, 0)
        y, featue = self.model.run([inp])
        return y, featue, ratio_h

class TextDet():
    def __init__(self,
                 input_shape: tuple=(192, 192),
                 triton_host: str="0.0.0.0:8001"):
        self.craft = CraftTextDet(input_shape= input_shape, triton_host= triton_host)
        self.refine = RefiTextDet(input_shape= input_shape, triton_host= triton_host)
    def infer(self, img):
        y, featue, ratio_h = self.craft.infer(img)
        score_link = self.refine.infer(y, featue)
        ratio_w = ratio_h
        score_text = y[0,:,:,0]
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, 0.5, 0.4, 0.4, False)
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        # polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        # for k in range(len(polys)):
            # if polys[k] is None: polys[k] = boxes[k]
        # bboxes_xxyy = []
        # h,w,c = img.shape
        return boxes, polys
    def align_image(self, image):
        bboxes, polys = self.infer(image)

        bboxes_xxyy = []
        h,w,c = image.shape
        ratios = []
        
        for box in bboxes:
            x_min = max(int(min(box, key=lambda x: x[0])[0]),1)
            x_max = min(int(max(box, key=lambda x: x[0])[0]),w-1)
            y_min = max(int(min(box, key=lambda x: x[1])[1]),3)
            y_max = min(int(max(box, key=lambda x: x[1])[1]),h-2)    

            if (x_max-x_min) > 15:
                ratio = (y_max-y_min)/(x_max-x_min)
                ratios.append(ratio)
            
        mean_ratio = np.mean(ratios) 

        if mean_ratio>=1:   
            image,bboxes = rotate_box(image,bboxes,None,True,False)
        
            bboxes, polys = self.infer(image)

        for box in bboxes:
            x_min = max(int(min(box, key=lambda x: x[0])[0]),1)
            x_max = min(int(max(box, key=lambda x: x[0])[0]),w-1)
            y_min = max(int(min(box, key=lambda x: x[1])[1]),3)
            y_max = min(int(max(box, key=lambda x: x[1])[1]),h-2)    
            bboxes_xxyy.append([x_min-1,x_max,y_min-1,y_max])
        return bboxes_xxyy

if __name__ =="__main__":
    craft = CraftTextDet()
    refine = RefiTextDet()
    import time
    t1 = time.time()
    img =cv2.imread('test.jpg')
    y, featue, ratio_h = craft.infer(img)
    score_link = refine.infer(y, featue)
    ratio_w = ratio_h
    score_text = y[0,:,:,0]
    # print(score_text.shape)
    # score_link = y[0,:,:,1]
    # score_link = y_refiner[0,:,:,0].cpu().data.numpy()
    print(score_link.shape)
    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, 0.5, 0.4, 0.4, False)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    bboxes_xxyy = []
    h,w,c = img.shape
    ratios = []

    for box in boxes:
        x_min = max(int(min(box, key=lambda x: x[0])[0]),1)
        x_max = min(int(max(box, key=lambda x: x[0])[0]),w-1)
        y_min = max(int(min(box, key=lambda x: x[1])[1]),3)
        y_max = min(int(max(box, key=lambda x: x[1])[1]),h-2)    
        bboxes_xxyy.append([x_min-1,x_max,y_min-1,y_max])

    if len(bboxes_xxyy) >0:
        for idx, text_box in enumerate(bboxes_xxyy):
            text_in_cell = img[text_box[2]:text_box[3], text_box[0]:text_box[1]]
            cv2.imwrite('result/'+str(idx)+'.jpg', text_in_cell)

    print(time.time() - t1)