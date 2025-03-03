 CUDA_VISIBLE_DEVICES=3 python tools/infer_rec.py \
  -c configs/rec/rec_r34_vd_none_bilstm_ctc.yml \
  -o Global.infer_img="./dataset/detection/Patches_rotate/Luc Van Tien/nlvnpf-0059-002_1.jpg" Global.pretrained_model="./output/rec/r34_vd_none_bilstm_ctc/best_accuracy"
