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

### Hardware optimizations: quantization and batch size. Optimize models with TensorRT.
Scripts folder :

    trt_conversion.sh
    
### Software optimizations: filter pruning. 
Remove filters (mains folder):
* 2D-CNN and 3D-CNN: train_singlenet_percentil.py
* GaitSet: gaitset_filters_percentil.py

Fine-tuning (mains folder):
* 2D-CNN and 3D-CNN: train_singlenet.py

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
