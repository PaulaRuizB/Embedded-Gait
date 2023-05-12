import tensorflow as tf
import numpy as np
from tensorflow_model_optimization.python.core.sparsity.keras.prune import strip_pruning


class countszeros(tf.keras.callbacks.Callback):
    def __init__(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        w = 0
        zeros = 0
        #print("He llegado a real target sparsity")
        self.model = strip_pruning(self.model)
        #self.model.summary()

        for layer in range(len(self.model.layers)):
            if len(self.model.layers[layer].weights) > 0:
                for i in range(len(self.model.layers[layer].weights)):
                    w = w + self.model.layers[layer].weights[i].numpy().size
                    zeros = zeros + np.count_nonzero(np.abs(self.model.layers[layer].weights[i].numpy()) < 1e-3)

        pct = 100 * float(zeros) / float(w)
        print("\n Real target sparsity {:.3f} ".format(pct))

