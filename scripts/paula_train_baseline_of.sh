python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --prefix of_baseline --nclasses 150 --use3d --epochs 150 --extraepochs 50
python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --prefix of_baseline --nclasses 155 --epochs 40 --extraepochs 10 --use3d --initnet /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of3D_opSGD_bs150_lr0.010000_dr0.40/model-state-0115.hdf5 --freezeall
python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --prefix of_baseline --nclasses 16 --epochs 40 --extraepochs 10 --use3d --initnet /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of3D_opSGD_bs150_lr0.010000_dr0.40/model-final.hdf5 --freezeall


# prueba 2D
python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --prefix prueba --nclasses 150 --epochs 150 --extraepochs 50

#python ../mains/test_singlenet.py --use3d --nclasses 155 --model /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N155_datagen_of3D_opSGD_frall_bs150_lr0.010000_dr0.40/model-final.hdf5
#python ../mains/test_singlenet.py --use3d --nclasses 16 --model /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N016_datagen_of3D_opSGD_frall_bs150_lr0.010000_dr0.40/model-final.hdf5
#python ../mains/test_singlenet_knn.py --use3d --nclasses 155 --knn 7 --model /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of3D_opSGD_bs150_lr0.010000_dr0.40/model-final.hdf5
#python ../mains/test_singlenet_knn.py --use3d --nclasses 16 --knn 7 --model /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of3D_opSGD_bs150_lr0.010000_dr0.40/model-final.hdf5

python ../mains/test_singlenet_knn.py --nclasses 155 --knn 7 --model /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of_opSGD_bs150_lr0.010000_dr0.40/model-final.hdf5
#######################################
##Pruning estructural load model
#PARA SACAR EL MODELO SIN FILTROS 3D 150
python ../mains/train_singlenet_percentil.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --prefix of_baseline --nclasses 150 --use3d --epochs 1 --extraepochs 50 --pstruc
#PARA SACAR EL MODELO SIN FILTROS 2D 150
python ../mains/train_singlenet_percentil.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --prefix of_baseline --nclasses 150 --epochs 1 --extraepochs 50 --pstruc
#TEST 3D 150
python ../mains/test_singlenet_knn.py --use3d --nclasses 16 --knn 7 --model /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of3D_opSGD_bs150_lr0.010000_dr0.40/model_without_49_filters.h5
#2d 150
python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --prefix of_baseline --nclasses 150 --epochs 1 --extraepochs 50 --pstruc


#MODELO 155 3D
python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --prefix of_baseline --nclasses 155 --epochs 40 --extraepochs 10 --use3d --initnet /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of3D_opSGD_bs150_lr0.010000_dr0.40/model-state-0115.hdf5 --freezeall --pstruc
#MODELO 155 2D
python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --prefix of_baseline --nclasses 155 --epochs 40 --extraepochs 10 --initnet /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of_opSGD_bs150_lr0.010000_dr0.40/model-state-0115.hdf5 --freezeall --pstruc

#TEST 3D 155
python ../mains/test_singlenet.py --use3d --nclasses 155 --model /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N155_datagen_of3D_opSGD_frall_bs150_lr0.010000_dr0.40/model-final.hdf5

#########################

#con el otro pruning antiguo sin 3d
python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --prefix of_baseline --nclasses 150 --pruning --target_sparsity 0.9 --frequency 100 --begin_step 1000 --epochs 1 --extraepochs 50 --pstruc

python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --prefix of_baseline --nclasses 155 --epochs 40 --extraepochs 10 --use3d --initnet /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of3D_opSGD_bs150_lr0.010000_dr0.40/model-state-0115.hdf5 --freezeall
python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --prefix of_baseline --nclasses 16 --epochs 40 --extraepochs 10 --use3d --initnet /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of3D_opSGD_bs150_lr0.010000_dr0.40/model-final.hdf5 --freezeall

##########FINE TUNING DE PRUNING #####
##Cambiar nombre modelo y percentil del prefix
#FT 2D
python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --prefix ft_pruning_80 --nclasses 150 --epochs 90 --extraepochs 10 --initnet /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of_opSGD_bs150_lr0.010000_dr0.40/model_without_1048_filters_80_41.13.h5 --ftpruning --nofreeze --lr 0.001 #prueba lenta
python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --prefix ft_pruning_10 --nclasses 150 --epochs 25 --extraepochs 5 --initnet /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of_opSGD_bs150_lr0.010000_dr0.40/model_without_132_filters_10_27.85.h5 --ftpruning --nofreeze
python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --prefix ft_pruning_20 --nclasses 150 --epochs 25 --extraepochs 5 --initnet /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of_opSGD_bs150_lr0.010000_dr0.40/model_without_263_filters_20_29.26.h5 --ftpruning --nofreeze
python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --prefix ft_pruning_30 --nclasses 150 --epochs 25 --extraepochs 5 --initnet /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of_opSGD_bs150_lr0.010000_dr0.40/model_without_394_filters_30_30.95.h5 --ftpruning --nofreeze

#FT 3D
python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --use3d --prefix ft_pruning_10_bueno --nclasses 150 --epochs 100 --extraepochs 10 --initnet /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of3D_opSGD_bs150_lr0.010000_dr0.40/model_without_198_filters_10_28.22.h5 --ftpruning --nofreeze
python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --use3d --prefix ft_pruning_20_bueno --nclasses 150 --epochs 90 --extraepochs 10 --initnet /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of3D_opSGD_bs150_lr0.010000_dr0.40/model_without_396_filters_20_29.97.h5 --ftpruning --nofreeze
python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --use3d --prefix ft_pruning_30_bueno --nclasses 150 --epochs 180 --extraepochs 10 --initnet /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of3D_opSGD_bs150_lr0.010000_dr0.40/model_without_594_filters_30_33.36.h5 --ftpruning --nofreeze #puesto
python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --use3d --prefix ft_pruning_40_bueno --nclasses 150 --epochs 90 --extraepochs 10 --initnet /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of3D_opSGD_bs150_lr0.010000_dr0.40/model_without_792_filters_40_49.44.h5 --ftpruning --nofreeze --lr 0.001 #se va
python ../mains/train_singlenet.py --model_version bmvc --experdir /home/pruiz/experiments_gait_multimodal/ --use3d --prefix ft_pruning_50_bueno --nclasses 150 --epochs 30 --extraepochs 10 --initnet /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of3D_opSGD_bs150_lr0.010000_dr0.40/model_without_990_filters_50_62.91.h5 --ftpruning --nofreeze --lr 0.001 #se va


#test 3d
python ../mains/test_singlenet_knn.py --use3d --nclasses 155 --knn 7 --model /home/pruiz/experiments_gait_multimodal/ft_pruning_bmvc_N150_datagen_of3D_opSGD_bs150_lr0.010000_dr0.40/model-state-0030.hdf5
#test 2d
python ../mains/test_singlenet_knn.py --nclasses 155 --knn 7 --model /home/pruiz/experiments_gait_multimodal/ft_pruning_50_bmvc_N150_datagen_of_opSGD_bs150_lr0.010000_dr0.40/model-final.hdf5
python ../mains/test_singlenet_knn.py --nclasses 155 --knn 7 --model /home/pruiz/experiments_gait_multimodal/ft_pruning_60_bmvc_N150_datagen_of_opSGD_bs150_lr0.010000_dr0.40/model-final.hdf5
python ../mains/test_singlenet_knn.py --nclasses 155 --knn 7 --model /home/pruiz/experiments_gait_multimodal/ft_pruning_70_bmvc_N150_datagen_of_opSGD_bs150_lr0.010000_dr0.40/model-final.hdf5
python ../mains/test_singlenet_knn.py --nclasses 155 --knn 7 --model /home/pruiz/experiments_gait_multimodal/ft_pruning_80_bmvc_N150_datagen_of_opSGD_bs150_lr0.001000_dr0.40/model-final.hdf5

#test original 3d
python ../mains/test_singlenet_knn.py --use3d --nclasses 155 --knn 7 --model /home/pruiz/experiments_gait_multimodal/of_baseline_bmvc_N150_datagen_of3D_opSGD_bs150_lr0.010000_dr0.40/model-state-0115.hdf5


