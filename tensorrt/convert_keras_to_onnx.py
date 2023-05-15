from tensorflow.python.keras import backend as K
import tensorflow as tf
import argparse
import os
from pathlib import Path
HUBER_DELTA = 0.5
import numpy as np


def mj_smoothL1(y_true, y_pred):
    x = K.abs(y_true - y_pred)
    x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return K.sum(x)


class MatMul(tf.keras.layers.Layer):
    def __init__(self, bin_num=31, hidden_dim=256, **kwargs):
        super(MatMul, self).__init__(**kwargs)

        self.bin_num = bin_num
        self.hidden_dim = hidden_dim

        # Create a trainable weight variable for this layer.
        w_init = tf.keras.initializers.GlorotUniform()
        self.kernel = tf.Variable(name="MatMul_kernel" + str(np.random.randint(100, size=1)[0]), ##[0]
                                  initial_value=w_init(shape=(bin_num*2, 128, hidden_dim), dtype="float32"),
                                  trainable=True)

    def call(self, x):
        # Implicit broadcasting occurs here.
        # Shape x: (BATCH_SIZE, N, M)
        # Shape kernel: (N, M)
        # Shape output: (BATCH_SIZE, N, M)
        return tf.matmul(x, self.kernel)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'bin_num': self.bin_num,
            'hidden_dim': self.hidden_dim,
        })
        return config


if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser(description='Evaluates a detection pipeline')
    parser.add_argument('--model_path_gait', type=str, default='', required=False, help="Path original model gait")
    parser.add_argument('--name_model', type=str, default='', required=False, help="Name model keras")
    parser.add_argument('--model_path_save', type=str, default='', required=True, help="Path to save model")
    args = parser.parse_args()
    model_path_gait = args.model_path_gait
    name_model = args.name_model
    model_path_save = args.model_path_save

    if name_model == 'mobilenet_v2':
        model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', pooling='avg')
    elif name_model == 'mobilenet':
        model = tf.keras.applications.mobilenet.MobileNet(include_top=False, weights='imagenet', pooling='avg')
    elif name_model == 'xception':
        model = tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', pooling='avg')
    else:
        model = tf.keras.models.load_model(model_path_gait,
                                           custom_objects={'mj_smoothL1': mj_smoothL1, 'MatMul': MatMul(), 'tf': tf},
                                           compile=False)  #
        if name_model == '':
            head_path, name_model = os.path.split(model_path_gait)
            name_model = Path(name_model).stem

    tf.saved_model.save(model, model_path_save + "/{}".format(name_model))
    print("Model created ok")
