## From TensorFlow to TensorRT

# From tenforflow to keras saved model
python3 convert_keras_to_onnx.py --model_path_gait /home/pruiz/Embedded-Gait-main/Embedded-Gait/models/model_2D_150.hdf5 --name_model model_2D_150 --model_path_save /home/pruiz/prueba_chema

# From keras saved model to ONNX
python3 -m tf2onnx.convert --saved-model '/home/pruiz/prueba_chema/model_2D_150' --output '/home/pruiz/prueba_chema/model_2D_150.onnx'
