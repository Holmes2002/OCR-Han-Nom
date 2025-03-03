## Vu model
# CUDA_VISIBLE_DEVICES=1 python tools/infer_det.py \
#   -c configs/det/det_r50_vd_east.yml \
#   -o Global.infer_img="/home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/dataset/detection/Pages/Handwritten_dataset/10/BNTwEHieafWla.1.87.jpg" \
#   Global.pretrained_model="output/east_r50_vd/latest"
CUDA_VISIBLE_DEVICES=1 python tools/infer_det.py \
  -c configs/det/det_r50_vd_east.yml \
  -o Global.infer_img="/home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/dataset/detection/Pages/Handwritten_dataset/10/BNTwEHieafWla.1.87.jpg" \
  Global.pretrained_model="output/det_r50_vd/best_accuracy"

## Thai Model
# CUDA_VISIBLE_DEVICES=5 python tools/infer_det.py \
#    -c /home1/vudinh/NomNaOCR/weights/Text_detection/EAST/config.yml \
#    -o Global.infer_img="/home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/dataset/detection/Pages/Luc Van Tien/imgs/nlvnpf-0059-003.jpg" \
#    Global.pretrained_model="/home1/vudinh/NomNaOCR/weights/Text_detection/EAST/best_accuracy"