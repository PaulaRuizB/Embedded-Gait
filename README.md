# Embedded-Gait
 Efficient implementation of gait recognition models on embedded systems
 
### Prerequisites
1. Clone this repository with git
```
git clone https://github.com/PaulaRuizB/Embedded-Gait
```
2. What you need to use the codes 

* 2D-CNN and 3D-CNN: 
Python 3.6.8 and requirements TUM_requirements.txt into venv_requirements folder.
* GaitSet:
Python 3.8 and requirements GaitSet_requirements.txt into venv_requirements folder.

3. Modify the codes according to your experiments. 

 Search #todo in train_singlenet.py (baseline experiment) and train_singlenet_percentil.py (experiments with pruning).

 * Modify percentile to remove filters. 
 * In pruning mode: if you are using 2D-CNN, change the name of last dense layer before break the sequence (line 189).
 * If you are training or using 2D-CNN, change last dense layer name in single_model.py line 374 to "code_1".
 * If you are training a model with 150 classes, remove last dense layer for KNN. If you are training 155 classes, let last dense layer.

### Baseline Models
Models folder:
* 2D-CNN: model_2D_150.hdf5
* 3D-CNN: model_3D_150.hdf5
* GaitSet: model-state-1300.hdf5 
### Train Models:
* 2D-CNN: 

      python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --prefix prueba --nclasses 150 --epochs 150 --extraepochs 50

* 3D-CNN:

      python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --prefix of_baseline --nclasses 150 --use3d --epochs 150 --extraepochs 50

### Test Models KNN:

Note: to measure accuracy comment EnergyMeter in encodeData function and set nbatches=len(data_generator)

* 2D-CNN: 

      python ../mains/test_singlenet_knn.py --nclasses 155 --knn 7 --model /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of_opSGD_bs150_lr0.010000_dr0.40/model-final.hdf5

* 3D-CNN:
    
      python ../mains/test_singlenet_knn.py --use3d --nclasses 155 --knn 7 --model /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of3D_opSGD_bs150_lr0.010000_dr0.40/model-final.hdf5

* GaitSet:

      python3 test_knn_gaitset_new.py --datadir=/home/GAIT_local/SSD_grande/CASIAB_tf_pre/ --model_version=gaitset_4dims --reshape_output --mod silhouette --bs 1 --knn 1 --nclasses 50 --allcameras --model /home/pruiz/gaitset_without_filters/model_gaitset_without_83_filters_10_14.95.h5 --cut --nframes=30

### Hardware optimizations: quantization, batch size and deep learning accelerators. Optimize models with TensorRT.
Scripts folder :

    trt_conversion.sh
    
### Software optimizations: filter pruning. 
Remove filters (mains folder):
* 2D-CNN and 3D-CNN: train_singlenet_percentil.py
 
Important: modify desired percentil value in code train_singlenet_percentil.py

2D-CNN:

    python ../mains/train_singlenet_percentil.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --prefix of_baseline --nclasses 150 --epochs 1 --extraepochs 50 --pstruc
  
3D-CNN: 

    python ../mains/train_singlenet_percentil.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --prefix of_baseline --nclasses 150 --use3d --epochs 1 --extraepochs 50 --pstruc

* GaitSet: gaitset_filters_percentil.py

Important: modify desired percentil value in code gaitset_filters_percentil.py

Fine-tuning (mains folder):
* 2D-CNN and 3D-CNN: train_singlenet.py

2D-CNN:

    python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --prefix ft_pruning_10 --nclasses 150 --epochs 25 --extraepochs 5 --initnet /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of_opSGD_bs150_lr0.010000_dr0.40/model_without_132_filters_10_27.85.h5 --ftpruning --nofreeze

3D-CNN:

    python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --use3d --prefix ft_pruning_10_bueno --nclasses 150 --epochs 100 --extraepochs 10 --initnet /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of3D_opSGD_bs150_lr0.010000_dr0.40/model_without_198_filters_10_28.22.h5 --ftpruning --nofreeze

### Our [paper](https://www.sciencedirect.com/science/article/pii/S2210537922001457)

If you find this code useful in your research, please consider citing:

    @article{ruiz2022high,
    title={High performance inference of gait recognition models on embedded systems},
    author={Ruiz-Barroso, Paula and Castro, Francisco M and Delgado-Esca{\~n}o, Rub{\'e}n and Ramos-C{\'o}zar, Juli{\'a}n and Guil, Nicol{\'a}s},
    journal={Sustainable Computing: Informatics and Systems},
    volume={36},
    pages={100814},
    year={2022},
    publisher={Elsevier}
    }
