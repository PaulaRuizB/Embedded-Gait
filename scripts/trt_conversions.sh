## From TensorFlow to TensorRT

# From tenforflow to keras saved model
python3 convert_keras_to_onnx.py --model_path_gait /path/model.hdf5 --name_model model --model_path_save /path

# From keras saved model to ONNX
python -m tf2onnx.convert --saved-model '/path/model' --output '/path/model.onnx' --opset 13

#Quantization
#From ONNX model to TensorRT engine + inference process.

#FP32
/usr/src/tensorrt/bin/trtexec --onnx=/path/model.onnx --saveEngine=/path/model_batch1_fp32.trt
/usr/src/tensorrt/bin/trtexec --loadEngine=/path/model_batch1_fp32.trt

#FP16
/usr/src/tensorrt/bin/trtexec --onnx=/path/model.onnx --saveEngine=/path/model_batch1_fp16.trt --fp16
/usr/src/tensorrt/bin/trtexec --loadEngine=/path/model_batch1_fp16.trt --fp16

#INT8
/usr/src/tensorrt/bin/trtexec --onnx=/path/model.onnx --saveEngine=/path/model_batch1_int8.trt --int8
/usr/src/tensorrt/bin/trtexec --loadEngine=/path/model_batch1_int8.trt --int8

#Best
/usr/src/tensorrt/bin/trtexec --onnx=/path/model.onnx --saveEngine=/path/model_batch1_best.trt --best
/usr/src/tensorrt/bin/trtexec --loadEngine=/path/model_batch1_best.trt --best

#Change batch size
#FP32
/usr/src/tensorrt/bin/trtexec --onnx=/path/model.onnx --saveEngine=/path/model_batch8_fp32.trt --shapes=\'input\':8x50x60x60
/usr/src/tensorrt/bin/trtexec --loadEngine=/path/model_batch8_fp32.trt

#DLA 1 and GPU
/usr/src/tensorrt/bin/./trtexec --onnx=/path/model.onnx --saveEngine=/path/model_dla1_batch1_fp32.trt --useDLACore=1 --allowGPUFallback