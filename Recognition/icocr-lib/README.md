# ICOCR - End2End Image2Text

<a href="https://pypi.org/project/icocr/0.0.1/"><img alt="Alt text" src="https://img.shields.io/badge/PyPI-3775A9.svg?style=for-the-badge&logo=PyPI&logoColor=white"/></a>

## Installation
### Install using pip:
```
pip install icocr
```
### Install from source:
```
git clone https://git.icomm.vn/thai.tran/ICOCR.git
pip install ./icocr-lib
```
### Download config and weights ONNX file from:
| File name | link download |
|---| ----- |
| Config | [link download](https://drive.google.com/file/d/1NrHoPO5boaDDNcd579bK5iHD7guYb7wY/view?usp=drive_link) |
| text_det_craft | [link download](https://drive.google.com/file/d/1Tn5MUTyOUtRqQZSu_YjvVhiEkVPMI3ml/view?usp=drive_link) |
| text_det_refine | [link download](https://drive.google.com/file/d/1owsijdhNvodzXqE8ucZNAg69f7hjoMar/view?usp=drive_link) |
| text_recog_encoder | [link download](https://drive.google.com/file/d/1iA7jX3oJMqY90rOyYbD4NOe68Avw392d/view?usp=drive_link) |
| text_recog_decoder | [link download](https://drive.google.com/file/d/1hDRn8DcgxQFWQCJt6lM7NOetJj-VkUj_/view?usp=drive_link) |

- Update your batch-size, device, directory weights of text detection models and text recognition models

## Usage
### Example icocr inference image2text

Inference class in icocr params:
```
- config: config file (modified path model onnx)
- batch_size: batch_size infernce text recognition model (default: 1)
- device: device used load onnx model (default: cpu)
```
Code example:
```
from icocr import icocr
import cv2
ocr = icocr.Inference(config_path=config.yml, batch_size=8, device='cuda')
img = cv2.imread("./icocr-lib/icocr/test_img.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #Input icocr is RGB image
out = ocr.icocr_predict(img)   #Dictionary of bounding box text and text prediction
```
| Image | icocr result |
| --- | ----- |
| <p align="center"> <img src="figs/test_img.png" width="960"> </p> | ({0: [[78, 12], [735, 3], [736, 31], [78, 39]], 1: [[17, 41], [734, 37], [734, 63], [17, 68]], 2: [[15, 72], [733, 69], [733, 95], [16, 98]], 3: [[18, 105], [634, 99], [634, 128], [18, 133]]}, {'0': 'Các đơn có nội dung chính sau: Tố cáo hành vi vi phạm pháp luật của', '1': 'Công ty TNHH Mặt trời Phú Quốc và các hành vi vi phạm quy định trong quân', '2': 'lý đất đai của một số cán bộ, lãnh đạo, các cơ quan chức năng liên quan trong', '3': 'quá trình thu hồi đất, giao đất cho Công ty TNHH Mặt trời Phú Quốc.'}) |

## Implementation Roadmap
The idea is to be able to plug in any state-of-the-art model into iCOCR.

<p align="center"> <img src="figs/icocr.jpg" width="960"> </p>

## Github Issues
Due to limited resources, an issue older than 6 months will be automatically closed. Please open an issue again if it is critical.

## Business Inquiries
For Enterprise Support, [iCOMM](https://icomm.vn/) offers full service for custom OCR/AI systems from implementation, training/finetuning and deployment. Click [here](contact@icomm.vn) to contact us.
