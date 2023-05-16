import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

import os.path as osp
from os.path import expanduser

import pathlib

maindir = pathlib.Path(__file__).parent.absolute()
if sys.version_info[1] >= 6:
    sys.path.insert(0, osp.join(maindir, ".."))
else:
    sys.path.insert(0, str(maindir) + "/..")
homedir = expanduser("~")
sys.path.insert(0, homedir + "/gaitmultimodal")
sys.path.insert(0, homedir + "/gaitmultimodal/mains")

theSEED = 232323
tf.random.set_seed(theSEED)
config = tf.compat.v1.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # gpu_rate

tf.executing_eagerly()
graph = tf.Graph()
graph.as_default()

session = tf.compat.v1.Session(graph=graph, config=config)
session.as_default()
# --------------------------------
from kerassurgeon.operations import delete_channels, delete_layer, insert_layer, Surgeon

# ===============================================================
#################Structural Pruning#################
####NOTAS:
# 1. Take into account the percentile to remove filters
# 2. Test: python3 test_knn_gaitset_new.py --datadir=/path/CASIAB/ --model_version=gaitset_4dims --reshape_output --mod silhouette --bs 1 --knn 1 --nclasses 50 --allcameras --model /path/model_gaitset_without_83_filters_10_14.95.h5 --cut --nframes=30

netpath = "/home/pruiz/model-state-1300.hdf5"
import tensorflow as tf
from nets.gaitset_4dims import GaitSet as OurModel
encode_layer = "encode"
model = OurModel("/home/pruiz/")
input_shape = (None, 64, 44, 1)
patch_size = 0
optimfun = tf.keras.optimizers.SGD()

model.build(input_shape=input_shape, optimizer=optimfun, margin=0.2, patchs_sizes=patch_size, attention_mode=None, crossentropy_weight=0)  # cross 1.0 # patchs_sizes=patch_size, attention_mode=None,crossentropy_weight=0
model.model.summary()
model.model.load_weights(netpath)
model.prepare_encode(encode_layer)

model = model.model
model.summary()

w_list = []
layer_name = []
values_total = []

for i in range(len(model.layers)):  # every layer in the model
    if 'keras.layers.Conv3D' in model.layers[i]._keras_api_names[0] or 'keras.layers.Conv2D' in \
            model.layers[i]._keras_api_names[0]:  # take convolutional layers
        w = model.layers[i].get_weights()[0]
        w_list.append(w)
        layer_name.append(model.layers[i].name)

for i in range(len(w_list)):
    weight = w_list[i]
    weight_dict = {}

    num_filters = len(weight[0, 0, 0, :])

    # L1-norm of every filter and save in a dictionary
    for j in range(num_filters):
        w_s = np.sum(abs(weight[:, :, :, j]))

        # filt = 'filt_{}'.format(j)
        weight_dict[j] = w_s

    values = np.fromiter(weight_dict.values(), dtype=float)
    values_total = np.append(values_total, values)

    # Filters in ascending L1 value
    weights_dict_sort = sorted(weight_dict.items(), key=lambda k: k[1])
    print("L1 norm conv layer {}\n".format(i + 1), weights_dict_sort)

values_total = np.sort(values_total)
value_percentile = 50  # TODO
percentile = np.percentile(values_total, value_percentile)

contador = 0
deleted = 0

for i in range(len(model.layers)):  # Sequential model
    if model.layers[i].name in layer_name:
        weight = w_list[contador]

        num_filters = len(weight[0, 0, 0, :])

        # L1-norm of every filter and save in a dictionary
        for j in reversed(range(num_filters)):

            w_s = np.sum(abs(weight[:, :, :, j]))

            if w_s < percentile and model.layers[i].output_shape[-1] > 1:
                try:
                    model = delete_channels(model, model.layers[i], np.array([j]))
                    deleted = deleted + 1
                except:
                    pass

        contador = contador + 1

print("{} Filters removed".format(deleted))

save_path = 'path/model_gaitset_without_{}_filters_{}_{:.2f}.h5'.format(deleted, value_percentile, percentile)
model.save(save_path)
print("Model with {} less filters saved".format(deleted))
