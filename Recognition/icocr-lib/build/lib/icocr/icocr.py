from icocr.text_detection import TextDet 
from icocr.text_recognition import TextRecog
from icocr.utils import crop_rect, preprocess_recog, Cfg
import cv2
import numpy as np
from scipy.spatial import distance
from PIL import Image
import os
def valid_xml_char_ordinal(c):
    codepoint = ord(c)
    # conditions ordered by presumed frequency
    return (
        0x20 <= codepoint <= 0xD7FF or
        codepoint in (0x9, 0xA, 0xD) or
        0xE000 <= codepoint <= 0xFFFD or
        0x10000 <= codepoint <= 0x10FFFF
        )

class Inference():
    def __init__(self, config_path: str, batch_size: int=1, device: str='cpu'):
        self.config = Cfg.load_config_from_file(config_path)
        self.device = device
        if 'cuda' in self.device:
            self.text_det = TextDet(self.config.text_det_craft, self.config.text_det_refine, True)
            self.text_reg = TextRecog(self.config.text_recog_encoder, self.config.text_recog_decoder, self.config.vocab, self.config.max_length_token, True)
        else:
            self.text_det = TextDet(self.config.text_det_craft, self.config.text_det_refine, False)
            self.text_reg = TextRecog(self.config.text_recog_encoder, self.config.text_recog_decoder, self.config.vocab, self.config.max_length_token, False)

        self.width_size = self.config.width_size
        self.height_size = self.config.height_size
        self.num_batch = batch_size
        if self.num_batch > 8:
            print("[WARNING] Max batch size recommend is 8")

    def icocr_predict(self, image):
        boxes, _ = self.text_det.inference(image)
        dct_box_text = dict()
        dct_text = dict()
        dct_box = dict()
        lst_text_in_box = list()
        text_result = list()

        for idx_box, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            poly = poly.reshape(-1, 2)
            v0 = np.array(poly[1]) - np.array(poly[0])
            v1 = np.array(poly[1]) - np.array([0,poly[1][1]])
            angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
            degree = 360-np.degrees(angle)
            width = max(distance.euclidean(poly[1], poly[0]), distance.euclidean(poly[3], poly[2]))
            height = max(distance.euclidean(poly[1], poly[2]), distance.euclidean(poly[3], poly[0]))
            size = (width, height)
            rect = cv2.minAreaRect(poly)
            im_crop, img_rot = crop_rect(image, rect, degree, size)

            ###
            os.makedirs('text_crop_fol', exist_ok = True)
            index = len(os.listdir('text_crop_fol'))
            cv2.imwrite(f'text_crop_fol/{index}.png', im_crop)
            ###
            im_crop = cv2.cvtColor(im_crop, cv2.COLOR_BGR2RGB)
            text_in_img = Image.fromarray(im_crop)
            lst_text_in_box.append(text_in_img)
            dct_box_text.update({idx_box: poly.tolist()})

        batch_array = preprocess_recog(lst_text_in_box, self.width_size, self.height_size)
        total_images = len(lst_text_in_box)
        total_batchs = int(total_images/self.num_batch) if total_images % self.num_batch == 0 else int(total_images/self.num_batch) + 1
        for ib in range(total_batchs):
            lower = ib * self.num_batch
            higher = min((ib+1)*self.num_batch, total_images)
            mini_batch = batch_array[lower:higher]
            output_text = self.text_reg.inference(pixel_values=mini_batch, max_seq_length=256, sos_token=1, eos_token=2)
            text_result.extend(output_text)

        for idx_box, text in enumerate(text_result):
            dct_text.update({str(idx_box): text})

        return dct_box_text, dct_text
    def pytesseract_predict(self, image):
        import pytesseract
        boxes, _ = self.text_det.inference(image)
        dct_box_text = dict()
        dct_text = dict()
        dct_box = dict()
        lst_text_in_box = list()
        text_result = list()

        for idx_box, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            poly = poly.reshape(-1, 2)
            v0 = np.array(poly[1]) - np.array(poly[0])
            v1 = np.array(poly[1]) - np.array([0,poly[1][1]])
            angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
            degree = 360-np.degrees(angle)
            width = max(distance.euclidean(poly[1], poly[0]), distance.euclidean(poly[3], poly[2]))
            height = max(distance.euclidean(poly[1], poly[2]), distance.euclidean(poly[3], poly[0]))
            size = (width, height)
            rect = cv2.minAreaRect(poly)
            im_crop, img_rot = crop_rect(image, rect, degree, size)
            im_crop = cv2.cvtColor(im_crop, cv2.COLOR_BGR2RGB)
            dct_box_text.update({idx_box: poly.tolist()})
            response_text = pytesseract.image_to_string(im_crop, lang="vie")
            cleaned_string = ''.join(c for c in response_text if valid_xml_char_ordinal(c))

            text_result.append(cleaned_string)
        for idx_box, text in enumerate(text_result):
            dct_text.update({str(idx_box): text})

        return dct_box_text, dct_text

if __name__ == "__main__":
    # config = Cfg.load_config_from_file("./univietocr.yml")

    infer = Inference("./config.yml", 4)
    img = cv2.imread("test1.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(infer.icocr_predict(img))
