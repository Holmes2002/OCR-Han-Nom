# Convert ICOCR2ONNX

## 1. Environment
```
torch==1.11.0+cu113
onnx==1.14.1
```

## 2. Convert ICOCR to ONNX
### 2.1 Convert encoder && decoder to ONNX
- Encoder converted:
```
python3 encoder2onnx.py
```
- Decoder converted:
```
python3 decoder2onnx.py
```
### 2.2 Inference ONNX model:
```
python3 inference_onnx.py
```

