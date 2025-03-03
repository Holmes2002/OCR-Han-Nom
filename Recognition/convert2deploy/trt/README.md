# Convert ICOCR to TensorRT

## 1. Environment
```
torch==1.11.0+cu113
onnx==1.14.1
tensorrt==8.6.1.post1
```

Install Pycuda:
```
git clone --recursive --branch v2020.1 https://github.com/inducer/pycuda.git
export PATH="/usr/local/cuda-11/bin:$PATH"
export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
nvcc --version
rm -rf site file
python configure.py --cuda-root=/usr/local/cuda-11
pip install -e .
```

## 2. Convert to ONNX

- Following [ICOCR2ONNX](../onnx/README.md)

## 3. Convert ICOCR to TensorRT
### 3.1 Convert encoder && decoder to TRT
- Encoder converted:
```
/usr/src/tensorrt/bin/trtexec --onnx=encoder_merge.onnx \
                                --saveEngine=encoder_merge.trt \
                                --explicitBatch \
                                --minShapes=input:1x3x32x128 \
                                --optShapes=input:4x3x32x128 \
                                --maxShapes=input:4x3x32x128 \
                                --verbose \
                                --device=1
```
- Decoder converted:
```
/usr/src/tensorrt/bin/trtexec --onnx=../onnx/models/decoder_merge.onnx \
                              --saveEngine=./modelsdecoder_merge.trt \
                              --explicitBatch \
                              --minShapes=tgt_inp:1x1,encoder_hidden_states:257x1x256 \
                              --optShapes=tgt_inp:8x8,encoder_hidden_states:257x8x256 \
                              --maxShapes=tgt_inp:128x8,encoder_hidden_states:257x8x256 \
                              --verbose \
                              --device=3
``` 

## 4. Inference ICOCR TRT

```
python3 utils.py
```