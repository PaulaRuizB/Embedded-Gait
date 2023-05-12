## From TensorFlow to TensorRT

# From tenforflow to keras saved model
python3 convert_keras_to_onnx.py --model_path_gait /home/pruiz/Embedded-Gait-main/Embedded-Gait/models/model_gaitset.hdf5 --name_model gaitset_keras --model_path_save /home/pruiz/prueba_chema

# From keras saved model to ONNX
!python -m tf2onnx.convert --saved-model '/home/pruiz/prueba_chema/gaitset_keras' --output '/home/pruiz/prueba_chema/model-state-1300.onnx'
