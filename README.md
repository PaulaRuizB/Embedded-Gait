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

### Optimize models with TensorRT (quantization and batch size)
Folder scripts

    trt_conversion.sh

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
