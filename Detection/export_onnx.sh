paddle2onnx --model_dir "./inference/en_PP-db_r50_OperSouce/" \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./inference/det_onnx/model_db_res50_OpenSource.onnx \
--opset_version 11 \
--enable_onnx_checker True 
