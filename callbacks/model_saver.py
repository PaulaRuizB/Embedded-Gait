import tensorflow as tf


class ModelSaver(tf.keras.callbacks.Callback):
	def __init__(self, model_object, **kwargs):
		super().__init__()
		self.model_object = model_object

	def on_epoch_end(self, epoch, logs=None):
		self.model_object.save(epoch)
