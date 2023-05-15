## From TensorFlow to TensorRT

# From tenforflow to keras saved model
python3 convert_keras_to_onnx.py --model_path_gait /home/pruiz/Embedded-Gait-main/Embedded-Gait/models/model_2D_150.hdf5 --name_model model_2D_150 --model_path_save /home/pruiz/prueba_chema

# From keras saved model to ONNX
python -m tf2onnx.convert --saved-model '/home/pruiz/prueba_chema/model_2D_150' --output '/home/pruiz/prueba_chema/model_2D_150.onnx' --opset 13

#Quantization

#FP32
/usr/src/tensorrt/bin/trtexec --onnx=/home/pruiz/pruebas_chema/model_2D_150.onnx --saveEngine=/home/pruiz/pruebas_chema/model_2D_150_batch1_fp32.trt
/usr/src/tensorrt/bin/trtexec --loadEngine=/home/nano/trt_models/model_2D_150_batch1_fp32.trt

#FP16
/usr/src/tensorrt/bin/trtexec --onnx=/home/nano/resnet50_onnx.onnx --saveEngine=/home/nano/trt_models/resnet50_batch1_fp16.trt --fp16
/usr/src/tensorrt/bin/trtexec --loadEngine=/home/nano/trt_models/resnet50_batch1_fp16.trt --fp16

#INT8
/usr/src/tensorrt/bin/trtexec --onnx=/home/nano/resnet50_onnx.onnx --saveEngine=/home/nano/trt_models/resnet50_batch1_int8.trt --int8
/usr/src/tensorrt/bin/trtexec --loadEngine=/home/nano/trt_models/resnet50_batch1_int8.trt --int8

#Best
/usr/src/tensorrt/bin/trtexec --onnx=/home/nano/resnet50_onnx.onnx --saveEngine=/home/nano/trt_models/resnet50_batch1_best.trt --best
/usr/src/tensorrt/bin/trtexec --loadEngine=/home/nano/trt_models/resnet50_batch1_best.trt --best

#Change batch size
#FP32
/usr/src/tensorrt/bin/trtexec --onnx=/home/pruiz/pruebas_chema/model_2D_150.onnx --saveEngine=/home/pruiz/pruebas_chema/model_2D_150_batch8_fp32.trt --shapes=\'input\':8x50x60x60
/usr/src/tensorrt/bin/trtexec --loadEngine=/home/nano/trt_models/resnet50_batch8_fp32.trt

#DLA and GPU
/usr/src/tensorrt/bin/./trtexec --onnx=/home/pruiz/pruebas_chema/model_2D_150.onnx --saveEngine=/home/pruiz/pruebas_chema/model_2D_150_dla1_batch1_fp32.trt --useDLACore=1--allowGPUFallback