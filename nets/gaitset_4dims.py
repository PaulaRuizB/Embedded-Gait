import tensorflow as tf
import tensorflow.keras.layers as layers
from nets.triplet_loss_all_gaitset import triplet_loss
import numpy as np
import os

class MatMul(tf.keras.layers.Layer):
	def __init__(self, bin_num=31, hidden_dim=256, **kwargs):
		super(MatMul, self).__init__(**kwargs)

		self.bin_num = bin_num
		self.hidden_dim = hidden_dim

		# Create a trainable weight variable for this layer.
		w_init = tf.keras.initializers.GlorotUniform()
		self.kernel = tf.Variable(name="MatMul_kernel"+str(np.random.randint(100, size=1)),
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


class GaitSet():

	def __init__(self, experdir, batch_size=128):
			self.model = None
			self.model_encode = None
			self.hist = None
			self.experdir = experdir
			self.batch_size = batch_size

	def load(self, previous_model):
		self.model = tf.keras.models.load_model(previous_model,
								custom_objects={"MatMul": MatMul(),
								"triplet_loss": triplet_loss()})


	def build(self, optimizer, margin=0.2, input_shape=(30, 64, 44, 1), **kwargs):

		input_layer = layers.Input(shape=input_shape)
		number_temp = 30

		# branch_a = layers.TimeDistributed(prune_low_magnitude(layers.Conv2D(32, kernel_size=5, activation=None, padding='same', use_bias=False,
		#											   data_format='channels_last'), **pruning_params))(input_layer)

		branch_a = tf.reshape(input_layer, [-1, input_layer.shape.as_list()[2], input_layer.shape.as_list()[3],
		                                    input_layer.shape.as_list()[4]])

		branch_a = layers.Conv2D(32, kernel_size=5, activation=None, padding='same', use_bias=False,
		                                          data_format='channels_last')(branch_a)

		branch_a = layers.LeakyReLU()(branch_a)

		branch_a = layers.Conv2D(32, kernel_size=3, activation=None, padding='same', use_bias=False,
		                                          input_shape=input_shape, data_format='channels_last')(branch_a)
		branch_a = layers.LeakyReLU()(branch_a)
		branch_a = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(branch_a)

		branch_b = tf.reshape(branch_a, [-1, number_temp, branch_a.shape.as_list()[1], branch_a.shape.as_list()[2],
		                                 branch_a.shape.as_list()[3]])
		branch_b = layers.Lambda(lambda x: tf.math.reduce_max(x, axis=1))(branch_b)
		branch_b = layers.Conv2D(64, kernel_size=3, activation=None, padding='same', use_bias=False,
		                                          data_format='channels_last')(branch_b)
		branch_b = layers.LeakyReLU()(branch_b)

		branch_b = layers.Conv2D(64, kernel_size=3, activation=None, padding='same', use_bias=False,
		                                          data_format='channels_last')(branch_b)
		branch_b = layers.LeakyReLU()(branch_b)
		branch_b = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(branch_b)

		branch_a = layers.Conv2D(64, kernel_size=3, activation=None, padding='same', use_bias=False,
		                                          input_shape=input_shape, data_format='channels_last')(branch_a)
		branch_a = layers.LeakyReLU()(branch_a)

		branch_a = layers.Conv2D(64, kernel_size=3, activation=None, padding='same', use_bias=False,
		                                          input_shape=input_shape, data_format='channels_last')(branch_a)
		branch_a = layers.LeakyReLU()(branch_a)
		branch_a = layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(branch_a)

		branch_b_ = tf.reshape(branch_a,
		                       [-1, number_temp, branch_a.shape.as_list()[1], branch_a.shape.as_list()[2],
		                        branch_a.shape.as_list()[3]])
		branch_b_ = layers.Lambda(lambda x: tf.math.reduce_max(x, axis=1))(branch_b_)
		branch_b = layers.Add()([branch_b, branch_b_])
		branch_b = layers.Conv2D(128, kernel_size=3, activation=None, padding='same', use_bias=False,
		                                          data_format='channels_last')(branch_b)
		branch_b = layers.LeakyReLU()(branch_b)
		branch_b = layers.Conv2D(128, kernel_size=3, activation=None, padding='same', use_bias=False,
		                                          data_format='channels_last')(branch_b)
		branch_b = layers.LeakyReLU()(branch_b)

		branch_a = layers.Conv2D(128, kernel_size=3, activation=None, padding='same', use_bias=False,
		                                          input_shape=input_shape, data_format='channels_last')(branch_a)
		branch_a = layers.LeakyReLU()(branch_a)

		branch_a = layers.Conv2D(128, kernel_size=3, activation=None, padding='same', use_bias=False,
		                                          input_shape=input_shape, data_format='channels_last')(branch_a)
		branch_a = layers.LeakyReLU()(branch_a)
		branch_a = tf.reshape(branch_a,
		                      [-1, number_temp, branch_a.shape.as_list()[1], branch_a.shape.as_list()[2],
		                       branch_a.shape.as_list()[3]])
		branch_a = layers.Lambda(lambda x: tf.math.reduce_max(x, axis=1))(branch_a)

		branch_b = layers.Add()([branch_b, branch_a])

		# HPP
		feature = list()
		bin_num = [1, 2, 4, 8, 16]
		n, h, w, c = branch_b.shape
		print(branch_b.shape)
		for num_bin in bin_num:
			branch_a_ = layers.Reshape((num_bin, -1, c))(branch_a)
			branch_a_ = layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=2) + tf.math.reduce_max(x, axis=2))(
				branch_a_)
			feature.append(branch_a_)
			branch_b_ = layers.Reshape((num_bin, -1, c))(branch_b)
			branch_b_ = layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=2) + tf.math.reduce_max(x, axis=2))(
				branch_b_)
			feature.append(branch_b_)

		model = layers.Concatenate(axis=1)(feature)
		model = layers.Lambda(lambda x: tf.transpose(x, [1, 0, 2]))(model)
		model = MatMul(name="encode")(model)

		outputs = []
		outputs.append(model)
		losses = []
		losses.append(triplet_loss(margin=margin))
		weights = []
		weights.append(1.0)
		if 'crossentropy_weight' in kwargs and kwargs['crossentropy_weight'] > 0.0:

			if 'nclasses' in kwargs:
				nclasses = kwargs['nclasses']
			else:
				nclasses = 74

			model2 = tf.transpose(model, [1, 0, 2])
			shapes_ = model2.shape
			model2 = tf.reshape(model2, [-1, shapes_[1] * shapes_[2]])
			tf.ensure_shape(model2, [None, shapes_[1] * shapes_[2]])
			model2 = layers.Dense(nclasses)(model2)
			model2 = layers.Softmax(name="probs")(model2)
			outputs.append(model2)
			losses.append(tf.keras.losses.SparseCategoricalCrossentropy())
			weights.append(kwargs['crossentropy_weight'])

		self.model = tf.keras.Model(inputs=input_layer, outputs=outputs)
		self.model.compile(optimizer=optimizer, loss=losses, metrics=None, loss_weights=weights)
		self.model_encode = self.model


	def fit(self, epochs, callbacks, training_generator, validation_generator, current_step=0, validation_steps=None, encode_layer=None, steps_per_epoch=None):
		self.hist = self.model.fit(training_generator, validation_data=validation_generator, epochs=epochs,
								   callbacks=callbacks, validation_steps=validation_steps, initial_epoch=current_step,
								   verbose=2, steps_per_epoch=steps_per_epoch)

		self.prepare_encode(encode_layer)
		return len(self.hist.epoch)

	def prepare_encode(self, encode_layer=None):
		if encode_layer is None:
			out_layer = self.model.get_layer("code").output
		else:
			out_layer = self.model.get_layer(encode_layer).output

		self.model_encode = tf.keras.Model(self.model.input, out_layer)

	def predict(self, data, batch_size=128):
		pred = self.model.predict(data, batch_size=batch_size)
		return pred

	def encode(self, data, batch_size=128):
		features = self.model_encode.predict(data, batch_size=batch_size)

		features = tf.transpose(features, [1, 0, 2])
		shapes_ = tf.shape(features)
		features = tf.reshape(features, [-1, shapes_[1] * shapes_[2]])

		# Get the numpy matrix
		codes_norm = features.numpy()
		return codes_norm

	def save(self, epoch=None):
		if epoch is not None:
			self.model.save(os.path.join(self.experdir, "model-state-{:04d}.hdf5".format(epoch)), include_optimizer=False, save_traces=False)

			# Save in such a way that can be recovered from different Python versions
			self.model.save_weights(os.path.join(self.experdir, "model-state-{:04d}_weights.hdf5".format(epoch)))
		else:
			self.model.save(os.path.join(self.experdir, "model-final.hdf5"), include_optimizer=False, save_traces=False)

			# Save in such a way that can be recovered from different Python versions
			self.model.save_weights(os.path.join(self.experdir, "model-final_weights.hdf5"))