# ICOCR
## Get Started

### 1. ICOCR UNION
- setup your params training in file config config/univietocr.yml
- venv: 172.16.10.240/home/data2/thaitran/tmp_venv/venv_38
- checkpoints printed text: 172.16.10.240/home/data2/thaitran/Research/OCR/source/vietocr-finetune/vietocr/union/ckpt
- training: 
```
CUDA_VISIBLE_DEVICES=1 python3 train_unionvietocr.py
```

### 2. ICOCR TRANS
- modify file config (config/trvietocr.yml)
- training: 
```
train_trvietocr.py
```

### 3. Convert ICOCR UNION to ONNX, TensorRT:
- Converted torch model to ONNX: Following pipeline [ICOCR2ONNX](convert2deploy/onnx/README.md)
- Converted torch model to TensorRT: Following pipeline [ICOCR2TRT](convert2deploy/trt/README.md)
- Deploy Triton: Following pipeline [ICOCR-Triton](convert2deploy/triton/README.md)

### 4. Deploy with fast api:
```
CUDA_VISIBLE_DEVCIES=1 python3 app-trocr.py
```

### 5. ICOCR End2End lib
- Following pipeline [ICOCR End2End](icocr-lib/README.md)
