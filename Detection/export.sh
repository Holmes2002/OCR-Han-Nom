## Detection
# CUDA_VISIBLE_DEVICES=2 python3 tools/export_model.py -c configs/det/det_mv3_east.yml   -o Global.pretrained_model="output/east_mv3/best_accuracy"  Global.save_inference_dir=./inference/en_PP-east/
# CUDA_VISIBLE_DEVICES=1 python3 tools/export_model.py -c configs/det/det_r50_vd_east.yml   -o Global.pretrained_model="output/east_r50_vd/best_accuracy"  Global.save_inference_dir=./inference/en_PP-east_r50/
# CUDA_VISIBLE_DEVICES=1 python3 tools/export_model.py -c configs/det/det_r50_db++_icdar15.yml   -o Global.pretrained_model="output/det_r50_icdar15_640/best_accuracy"  Global.save_inference_dir=./inference/en_PP-db++_r50/
# CUDA_VISIBLE_DEVICES=1 python3 tools/export_model.py -c configs/det/det_mv3_db.yml   -o Global.pretrained_model="output/db_mv3/best_accuracy"  Global.save_inference_dir=./inference/en_PP-db_mbv3/
# CUDA_VISIBLE_DEVICES=1 python3 tools/export_model.py -c configs/det/det_r50_vd_db.yml   -o Global.pretrained_model="output/det_r50_vd/best_accuracy"  Global.save_inference_dir=./inference/en_PP-db-50/
CUDA_VISIBLE_DEVICES=1 python3 tools/export_model.py -c /home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/output/Newer_ver/det_r50_vd/config.yml   -o Global.pretrained_model="/home1/vudinh/NomNaOCR/PaddleOCR-release-2.6/output/Newer_ver/det_r50_vd/best_accuracy"  \
    Global.save_inference_dir=./inference/en_PP-db_r50_OperSouce/

#Recognition
# python3 tools/export_model.py -c configs/rec/rec_r34_vd_none_bilstm_ctc.yml -o Global.pretrained_model=./output/rec/r34_vd_none_bilstm_ctc/best_accuracy  Global.save_inference_dir=./inference/rec_starnet
