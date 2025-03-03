import argparse
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--img', required=True, help='foo help')
#     parser.add_argument('--config', required=True, help='foo help')

#     args = parser.parse_args()
#     config = Cfg.load_config_from_file(args.config)

#     detector = Predictor(config)

#     img = Image.open(args.img)
#     s = detector.predict(img)

#     print(s)


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--img', required=True, help='foo help')
    # parser.add_argument('--config', required=True, help='foo help')

    # args = parser.parse_args()
    config_dir = "/home/data2/thaitran/Research/OCR/source/vietocr-finetune/vietocr/config/vgg-transformer.yml"
    config = Cfg.load_config_from_file(config_dir)
    config["device"] = "cuda"
    print(config)
    detector = Predictor(config)
    img_dir = "/home/data2/thaitran/Research/OCR/source/vietocr-finetune/vietocr/1.png"
    img = Image.open(img_dir)
    s = detector.predict(img)

    print(s)

if __name__ == '__main__':
    main()
