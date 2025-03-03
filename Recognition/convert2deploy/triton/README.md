# Deploy ICOCR on TRITON

## 1. Environment
```
nvcr.io/nvidia/tritonserver:22.12-py3
```

## 2. Push model to Triton
### 2.1 Convert to ONNX and TensorRT:
- following: [ICOCR2ONNX](../onnx/README.md) and [ICOCR2TRT](../trt/README.md)

### 2.2 Follow config in folder [config_triton](./models)

### 2.3 Inference Triton
- Modified model name, host trion in file [triton_backend.py](./exec_backend/triton_backend.py)
- Then, run:
```
python3 utils.py
```
