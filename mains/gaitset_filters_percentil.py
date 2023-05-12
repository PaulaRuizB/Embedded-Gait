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
# sys.path.insert(0, homedir + "/gaitmultimodal/nets")

# --------------------------------

theSEED = 232323
tf.random.set_seed(theSEED)
config = tf.compat.v1.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # gpu_rate # TODO

tf.executing_eagerly()
graph = tf.Graph()
graph.as_default()

session = tf.compat.v1.Session(graph=graph, config=config)
session.as_default()
# --------------------------------
from tensorflow import keras
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike

from kerassurgeon.operations import delete_channels, delete_layer, insert_layer, Surgeon

from nets.gaitset_4dims import MatMul


def triplet_loss(margin: FloatTensorLike = 1.0, k: TensorLike = 16):
    @tf.function
    def loss(y_true: TensorLike, y_pred: TensorLike) -> tf.Tensor:
        """Computes the triplet loss with semi-hard negative mining.

        Args:
          y_true: 1-D integer `Tensor` with shape [batch_size] of
            multiclass integer labels.
          y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
            be l2 normalized.
          margin: Float, margin term in the loss definition.

        Returns:
          triplet_loss: float scalar with dtype of y_pred.
        """
        labels, embeddings = y_true, y_pred

        convert_to_float32 = (
                embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
        )
        precise_embeddings = (
            tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
        )

        # n, m, d = tf.shape(embeddings)
        _shape = tf.shape(embeddings)
        n = _shape[0]
        m = _shape[1]
        d = _shape[2]
        # labels = tf.squeeze(labels)
        # labels = tf.repeat(tf.expand_dims(labels, axis=0), n, axis=0)
        labels = tf.transpose(labels, [1, 0])
        labels = tf.repeat(labels, n, axis=0)
        hp_mask = tf.reshape(tf.expand_dims(labels, axis=1) == tf.expand_dims(labels, axis=2), shape=[-1])
        hn_mask = tf.reshape(tf.expand_dims(labels, axis=1) != tf.expand_dims(labels, axis=2), shape=[-1])

        dist = batch_dist(embeddings)

        dist = tf.reshape(dist, shape=[-1])

        full_hp_dist = tf.reshape(tf.boolean_mask(dist, hp_mask), [n, m, -1, 1])
        full_hn_dist = tf.reshape(tf.boolean_mask(dist, hn_mask), [n, m, 1, -1])
        full_loss_metric = tf.reshape(tf.math.maximum(margin + tf.subtract(full_hp_dist, full_hn_dist), 0.0), [n, -1])

        full_loss_metric_sum = tf.math.reduce_sum(full_loss_metric, axis=1)
        valid_triplets = tf.cast(tf.greater(full_loss_metric, 0.0), dtype=tf.dtypes.float32)
        full_loss_num = tf.math.reduce_sum(valid_triplets, axis=1)

        full_loss_metric_mean = full_loss_metric_sum / full_loss_num

        zero = tf.constant(0.0, dtype=tf.float32)
        where = tf.not_equal(full_loss_num, zero)
        full_loss_metric_mean = tf.where(where, full_loss_metric_mean, tf.zeros_like(full_loss_metric_mean))

        full_loss_metric_mean = tf.reduce_mean(full_loss_metric_mean, axis=0)

        if convert_to_float32:
            return tf.cast(full_loss_metric_mean, embeddings.dtype)
        else:
            return full_loss_metric_mean

    return loss


# ===============================================================
#################Structural Pruning#################
####NOTAS:
# 1. Tener en cuenta el percentil de eliminación de filtros
# python3 test_knn_gaitset_new.py --datadir=/home/GAIT_local/SSD_grande/CASIAB_tf_pre/ --model_version=gaitset_4dims --reshape_output --mod silhouette --bs 1 --knn 1 --nclasses 50 --allcameras --model /home/pruiz/gaitset_without_filters/model_gaitset_without_83_filters_10_14.95.h5 --cut --nframes=30


######Crear modelo nuevo

netpath = "/home/pruiz/model-state-1300.hdf5"
import tensorflow as tf
# model = tf.keras.models.load_model(netpath, custom_objects={"MatMul": MatMul, "tf":tf}, compile=False)
from nets.gaitset_4dims import GaitSet as OurModel
#from nets.gaitset import GaitSet as OurModel
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

for i in range(len(model.layers)):  # recorre el modelo
    if 'keras.layers.Conv3D' in model.layers[i]._keras_api_names[0] or 'keras.layers.Conv2D' in \
            model.layers[i]._keras_api_names[0]:  # coge las conv
        w = model.layers[i].get_weights()[0]
        w_list.append(w)
        layer_name.append(model.layers[i].name)

for i in range(len(w_list)):
    weight = w_list[i]
    weight_dict = {}

    num_filters = len(weight[0, 0, 0, :])

    # Cálculo norma L1 de cada filtro de peso y se almacena en un diccionario
    for j in range(num_filters):
        w_s = np.sum(abs(weight[:, :, :, j]))

        # filt = 'filt_{}'.format(j)
        weight_dict[j] = w_s

    values = np.fromiter(weight_dict.values(), dtype=float)
    values_total = np.append(values_total, values)

    # Se muestran los filtros por su valor L1 ascendente
    weights_dict_sort = sorted(weight_dict.items(), key=lambda k: k[1])
    print("L1 norm conv layer {}\n".format(i + 1), weights_dict_sort)

values_total = np.sort(values_total)
valor_percentil = 50  # TODO
percentil = np.percentile(values_total, valor_percentil)

contador = 0
deleted = 0

for i in range(len(model.layers)):  # recorre sequential
    if model.layers[i].name in layer_name:
        weight = w_list[contador]

        num_filters = len(weight[0, 0, 0, :])

        # Cálculo norma L1 de cada filtro de peso y se almacena en un diccionario
        for j in reversed(range(num_filters)):

            w_s = np.sum(abs(weight[:, :, :, j]))

            if w_s < percentil and model.layers[i].output_shape[-1] > 1:
                try:
                    model = delete_channels(model, model.layers[i], np.array([j]))
                    deleted = deleted + 1
                except:
                    pass

        contador = contador + 1

print("{} Filtros eliminados".format(deleted))
# model_filters.summary()

ruta = '/home/pruiz/gaitset_without_filters/model_gaitset_without_{}_filters_{}_{:.2f}.h5'.format(deleted,
                                                                                                  valor_percentil,
                                                                                                  percentil)
model.save(ruta)
print("MODELO CON {} MENOS FILTROS GUARDADO".format(deleted))
