import tensorflow as tf
#import tensorflow_addons as tfa
import deepdish as dd
import os
#from nets.triplet_crossentropy_loss import TripletSemiHardCrossentropyLoss, TripletBatchAllCrossentropyLoss, TripletHardCrossentropyLoss
#from nets.triplet_loss_all import TripletBatchAllLoss
#from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
#from tensorflow_model_optimization.python.core.sparsity.keras import prune
#from tensorflow_model_optimization.python.core.sparsity.keras.pruning_wrapper import PruneLowMagnitude

class SingleGaitModel():
	def __init__(self, experdir, iters_per_update=0):
		self.model = None
		self.model_encode = None
		self.hist = None
		self.experdir = experdir
		self.iters_per_update = iters_per_update

	def load(self, netpath, encode_layer=None):
		try:
			self.model = tf.keras.models.load_model(netpath) #custom_objects={"TripletBatchAllCrossentropyLoss": TripletBatchAllCrossentropyLoss(), "TripletBatchAllLoss": TripletBatchAllLoss(), 'PruneLowMagnitude': PruneLowMagnitude}
			print("++++++++++++++++++++++++++++++++++++++++++++")
			print("Model loaded from: " + netpath)
		except:
			# Load config file and build model.
			bdir = os.path.dirname(netpath)
			fconfig = os.path.join(bdir, "model-config.hdf5")
			netconfig = dd.io.load(fconfig)
			self.model = self.build_by_config(netconfig)

			# Load weights file.
			bname = os.path.basename(netpath)
			fparts = os.path.splitext(bname)
			filewes = os.path.join(bdir, fparts[0] + "_weights.hdf5")
			self.model.load_weights(filewes, by_name=True)

			print("++++++++++++++++++++++++++++++++++++++++++++")
			print("Model loaded from config + weight files: " + fconfig + '     ' + filewes)

		if encode_layer is None:
			#print("+++++++++++++++++++++++++++++++++ CONCAT+++++++++++++++++++++")
			#self.model_encode = tf.keras.Model(self.model.input, self.model.get_layer("concatenate").output)
			self.model_encode = tf.keras.Model(self.model.input, self.model.layers[-1].input)
		else:
			out_layer = self.model.get_layer(encode_layer).output
			self.model_encode = tf.keras.Model(self.model.input, out_layer)

	def build_or_load(self, input_shape, number_convolutional_layers, filters_size, filters_numbers, strides,
	                  ndense_units=2048, weight_decay=0.0001, dropout=0.4, optimizer=tf.keras.optimizers.SGD(0.01, 0.9),
	                  nclasses=0, initnet="", freeze_convs=False, use3D=False, freeze_all=False, model_version='iwann', lstm=-1,
					  lstm_number=512, dropout_lstm=0.0, L2_norm_lstm=None, loss_mode='crossentropy', margin=0.25, loss_weights=[1.0, 0.1],
					  activation_fun='relu', pruning=False, frequency = 100, begin_step = 2000, target_sparsity = 0.9, from3dto2d=False,
					  batchsize=150):

		if initnet == "":
			self.build(input_shape, number_convolutional_layers, filters_size, filters_numbers, strides,
			           ndense_units, weight_decay, dropout, optimizer, nclasses, use3D=use3D, model_version=model_version, lstm=lstm,
					   lstm_number=lstm_number, dropout_lstm=dropout_lstm, L2_norm_lstm=L2_norm_lstm, loss_mode=loss_mode, margin=margin,
			           loss_weights=loss_weights, activation_fun=activation_fun, pruning=pruning, frequency=frequency, begin_step=begin_step,
							   target_sparsity=target_sparsity, from3dto2d=from3dto2d, batchsize=batchsize)
		else:
			self.load(initnet)

			# Check if freeze some weights
			if freeze_convs or freeze_all:
				for layer in self.model.layers:
					if freeze_all or type(layer) == tf.keras.layers.Conv2D or type(layer) == tf.keras.layers.Conv3D or type(layer) == tf.keras.layers.TimeDistributed:
						if len(layer.trainable_variables) > 0:
							layer.trainable = False
					elif not freeze_all and type(layer) == tf.keras.layers.Dense:
						if layer.name != 'code':
							layer.trainable = False
					elif type(layer) == tf.keras.Sequential:
						for layer2 in layer.layers:
							if len(layer2.trainable_variables) > 0:
								layer2.trainable = False

			# # Check if exists FC for classification:
			# replace_fc = False
			# for layer in self.model.layers:
			# 	if layer.name == "probs":
			# 		# Check number of classes.
			# 		if layer.units != nclasses:
			# 			print("Replacing FC layer for the new classes...")
			# 			replace_fc = True
			#
			# if replace_fc:
			# 	main_branch = self.model.layers[-1].input
			# 	main_branch = tf.keras.layers.Dense(nclasses, activation='softmax', kernel_initializer='he_uniform',
			# 	                                    name="probs")(main_branch)

			output = self.model.layers[-1].input
			outputs = []
			losses = []
			metrics = []

			if loss_mode == 'crossentropy':
				# Insert a classification layer here
				main_branch = tf.keras.layers.Dense(nclasses, activation='softmax', kernel_initializer='he_uniform',
				                                    name="probs")(output)
				outputs.append(main_branch)
				losses.append('sparse_categorical_crossentropy')
				metrics.append(tf.keras.metrics.SparseCategoricalAccuracy())
			elif loss_mode == 'triplet':
				main_branch = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(
					output)
				outputs.append(main_branch)
				losses.append(tfa.losses.TripletSemiHardLoss(margin=margin))
				metrics = None
			elif loss_mode == 'triplet_crossentropy':
				main_branch = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(
					output)
				outputs.append(main_branch)
				losses.append(tfa.losses.TripletSemiHardLoss(margin=margin))
				main_branch_ = tf.keras.layers.Dense(nclasses, activation='softmax', kernel_initializer='he_uniform', name="probs")(output)
				outputs.append(main_branch_)
				losses.append('sparse_categorical_crossentropy')
				metrics = None
			elif loss_mode == 'tripletall':
#				main_branch = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(
#					output)
#				outputs.append(main_branch)
				outputs.append(output)
				losses.append(TripletBatchAllLoss(margin=margin))
				metrics = None
			elif loss_mode == 'tripletallcross':
				main_branch1 = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(
					output)
				outputs.append(main_branch1)
				losses.append(TripletBatchAllLoss(margin=margin))
				main_branch = tf.keras.layers.Dense(nclasses, activation='softmax', kernel_initializer='he_uniform',
				                                    name="probs")(output)
				outputs.append(main_branch)
				losses.append('sparse_categorical_crossentropy')
				metrics = {'probs': tf.keras.metrics.SparseCategoricalAccuracy()}
			elif loss_mode == 'tripletboth':
				main_branch = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(
					output)
				outputs.append(main_branch)
				losses.append(tfa.losses.TripletSemiHardLoss(margin=margin))
				outputs.append(main_branch)
				losses.append(tfa.losses.TripletHardLoss(margin=margin))
				metrics = None
			elif loss_mode == 'triplet_drop':
				output = tf.keras.layers.Dropout(dropout, name="drop_code")(output)
				main_branch = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(
					output)
				outputs.append(main_branch)
				losses.append(tfa.losses.TripletSemiHardLoss(margin=margin))
				metrics = None
			elif loss_mode == 'crosstriplet':
				#			main_branch = output
				main_branch = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(
					output)
				outputs.append(main_branch)
				losses.append(TripletSemiHardCrossentropyLoss(margin=margin))
				metrics = None
			elif loss_mode == 'crosstriplet_nol2':
				main_branch = output
				outputs.append(main_branch)
				losses.append(TripletSemiHardCrossentropyLoss(margin=margin))
				metrics = None
			elif loss_mode == 'both_crosstriplet':
				main_branch_ = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(
					output)
				outputs.append(main_branch_)
				main_branch = tf.keras.layers.Dense(nclasses, activation='softmax', kernel_initializer='he_uniform',
				                                    name="probs")(output)
				outputs.append(main_branch)
				losses.append(TripletSemiHardCrossentropyLoss(margin=margin))
				losses.append('sparse_categorical_crossentropy')
				metrics = None
			elif loss_mode == 'both_batchall':
				main_branch_ = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(
					output)
				outputs.append(main_branch_)
				main_branch = tf.keras.layers.Dense(nclasses, activation='softmax', kernel_initializer='he_uniform',
				                                    name="probs")(output)
				outputs.append(main_branch)
				losses.append(TripletBatchAllCrossentropyLoss(margin=margin))
				losses.append('sparse_categorical_crossentropy')
				metrics = None
			elif loss_mode == 'both_batchall_drop':
				output = tf.keras.layers.Dropout(dropout, name="drop_code")(output)
				main_branch_ = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(
					output)
				outputs.append(main_branch_)
				main_branch = tf.keras.layers.Dense(nclasses, activation='softmax', kernel_initializer='he_uniform',
				                                    name="probs")(output)
				outputs.append(main_branch)
				losses.append(TripletBatchAllCrossentropyLoss(margin=margin))
				losses.append('sparse_categorical_crossentropy')
				metrics = None
			elif loss_mode == 'both_hardcros':
				main_branch_ = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(
					output)
				outputs.append(main_branch_)
				main_branch = tf.keras.layers.Dense(nclasses, activation='softmax', kernel_initializer='he_uniform',
				                                    name="probs")(output)
				outputs.append(main_branch)
				losses.append(TripletHardCrossentropyLoss(margin=margin))
				losses.append('sparse_categorical_crossentropy')
				metrics = None
			elif loss_mode == 'both_batchalldist':
				main_branch_ = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(
					output)
				outputs.append(main_branch_)
				main_branch = tf.keras.layers.Dense(nclasses, activation='softmax', kernel_initializer='he_uniform',
				                                    name="probs")(output)
				outputs.append(main_branch)
				losses.append(TripletBatchAllLoss(margin=margin))
				losses.append('sparse_categorical_crossentropy')
				metrics = None
			else:
				import sys
				sys.exit('Wrong losses')
			print("Alright")

			self.model = tf.keras.Model(inputs=self.model.input, outputs=outputs)
			self.model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

	def build_by_config(self, netconfig):
		filters_size = netconfig["filters_size"]
		filters_numbers = netconfig["filters_numbers"]
		#strides = [1,1,1,1]
		strides = netconfig["strides"]
		input_shape = netconfig["input_shape"]
		ndense_units = netconfig["ndense_units"]
		weight_decay = netconfig["weight_decay"]
		dropout = netconfig["dropout"]
		if "nclasses" in netconfig.keys():
			nclasses = netconfig["nclasses"]
		else:
			nclasses = 150
		optimizer = netconfig["optimizer"]

		if "use3D" in netconfig.keys():
			use3D = netconfig["use3D"]
		else:
			use3D = False

		model_version = netconfig["model_version"]
		lstm = netconfig["lstm"]
		lstm_number = netconfig["lstm_number"]
		dropout_lstm = netconfig["dropout_lstm"]
		L2_norm_lstm = netconfig["L2_norm_lstm"]

		self.model = self.build(input_shape, len(filters_numbers), filters_size, filters_numbers, strides, ndense_units,
		                        weight_decay, dropout, nclasses=nclasses, optimizer=optimizer, use3D=use3D, model_version=model_version,
		                        lstm=lstm, lstm_number=lstm_number, dropout_lstm=dropout_lstm, L2_norm_lstm=L2_norm_lstm)

	def build(self, input_shape, number_convolutional_layers, filters_size, filters_numbers, strides, ndense_units=512,
	          weight_decay=0.0005, dropout=0.4, optimizer=tf.keras.optimizers.SGD(0.01, 0.9), nclasses=0, use3D=False, model_version='iwann', lstm=-1,
			  lstm_number=512, dropout_lstm=0.0, L2_norm_lstm=None, loss_mode='crossentropy', margin=0.25, loss_weights=[1.0, 0.1], activation_fun='relu',
			  pruning=False, frequency=100, begin_step=2000, target_sparsity=0.9, from3dto2d=False, batchsize=150):
		"""
		        :param input_shapes: tuple ((50,60,60), (25,60,60))
		        :param number_convolutional_layers:
		        :param filters_size:
		        :param filters_numbers:
		        :param ndense_units:
		        :param weight_decay:
		        :param dropout:
		        :param optimizer:
		        :param margin:
		        :return:
		        """

		if number_convolutional_layers < 1:
			print("ERROR: Number of convolutional layers must be greater than 0")

		outputs = []
		losses = []
		metrics = []

		if lstm == 0:
			input = tf.keras.layers.Input(shape=(None, input_shape[0], input_shape[1], input_shape[2]), name='input')
			lstm_layer = tf.keras.layers.ConvLSTM2D(16, 3, padding='same', data_format='channels_first')(input)
			input_shape = (16, input_shape[1], input_shape[2])
		elif lstm == 1 or lstm == 2:
			input = tf.keras.layers.Input(shape=(None, input_shape[0], input_shape[1], input_shape[2]), name='input')
			input_shape = (None, input_shape[0], input_shape[1], input_shape[2])
		elif lstm == 5:
			input = tf.keras.layers.Input(shape=(None, input_shape[0], input_shape[1], input_shape[2]), name='input')
			lstm_layer = tf.keras.layers.ConvLSTM2D(50, 3, padding='same', data_format='channels_first')(input)
			input_shape = (50, input_shape[1], input_shape[2])
		elif lstm == 6:
			input = tf.keras.layers.Input(shape=(None, input_shape[0], input_shape[1], input_shape[2]), name='input')
			lstm_layer = tf.keras.layers.ConvLSTM2D(16, 7, padding='same', data_format='channels_first')(input)
			input_shape = (16, input_shape[1], input_shape[2])
		elif lstm == 7:
			input = tf.keras.layers.Input(shape=(None, input_shape[0], input_shape[1], input_shape[2]), name='input')
			lstm_layer = tf.keras.layers.ConvLSTM2D(50, 7, padding='same', data_format='channels_first')(input)
			input_shape = (50, input_shape[1], input_shape[2])
		elif lstm >= 8:
			input = tf.keras.layers.Input(shape=(None, input_shape[0], input_shape[1], input_shape[2]), name='input')
			#lstm_layer = tf.keras.layers.ConvLSTM2D(50, 3, padding='same', data_format='channels_first')(input)
			L2_norm = tf.keras.regularizers.l2(weight_decay)
			if activation_fun == 'leaky':
				lstm_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(25, kernel_size=3, kernel_regularizer=L2_norm, activation=None, data_format='channels_first', padding='same'))(input)
				lstm_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.LeakyReLU())(lstm_layer)
			else:
				lstm_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(25, kernel_size=3, kernel_regularizer=L2_norm, activation='relu', data_format='channels_first', padding='same'))(input)

			lstm_layer = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=1))(lstm_layer)
			input_shape = [(25, input_shape[1], input_shape[2]), (None, input_shape[0], input_shape[1], input_shape[2])]
		else:
			input = tf.keras.layers.Input(shape=input_shape, name='input')
		if use3D:
			if model_version == 'bmvc' and not from3dto2d:
				convBranch = self.build_3Dbranch_Manuel("convBranch", input_shape, ndense_units, pruning,
														  frequency, begin_step,target_sparsity)
			if model_version == 'bmvc' and from3dto2d:
				convBranch = self.branch3dto2d1d("convBranch", batchsize, input, input_shape, ndense_units, pruning,
														  frequency, begin_step,target_sparsity)
			#else:
			#	convBranch = self.build_3Dbranch("convBranch", input_shape, number_convolutional_layers, filters_size, strides,
			#	                           filters_numbers, weight_decay, ndense_units, dropout)
		else:
			if model_version == 'bmvc':
				if lstm >= 8:
					lstm_branch1 = 5
					lstm_branch2 = 2
					convBranch1 = self.build_branch_Manuel("convBranch", input_shape[0], number_convolutional_layers,
					                                      filters_size, strides, filters_numbers, weight_decay,  ndense_units,
					                                       dropout, lstm_branch1, lstm_number, dropout_lstm, L2_norm_lstm, False, activation_fun)
					convBranch2 = self.build_branch_Manuel("convBranch2", input_shape[1], number_convolutional_layers,
					                                       filters_size, strides, filters_numbers, weight_decay, ndense_units,
					                                       dropout, lstm_branch2, lstm_number, dropout_lstm, L2_norm_lstm, False, activation_fun)
					convBranch = [convBranch1, convBranch2]
				else:
					convBranch = self.build_branch_Manuel("convBranch", input_shape, number_convolutional_layers, filters_size,
				                               strides, filters_numbers, weight_decay, ndense_units, dropout, lstm, lstm_number, dropout_lstm, L2_norm_lstm, True, activation_fun,
														  pruning,frequency, begin_step,target_sparsity)
			else:
				if lstm >= 8:
					lstm_branch1 = 5
					lstm_branch2 = 2
					convBranch1 = self.build_branch("convBranch", input_shape[0], number_convolutional_layers,
					                                filters_size, strides, filters_numbers, weight_decay, ndense_units,
					                                dropout, lstm_branch1, lstm_number, dropout_lstm, L2_norm_lstm, False, activation_fun)
					convBranch2 = self.build_branch("convBranch2", input_shape[1], number_convolutional_layers,
					                                filters_size, strides, filters_numbers, weight_decay, ndense_units,
					                                dropout, lstm_branch2, lstm_number, dropout_lstm, L2_norm_lstm, False, activation_fun)
					convBranch = [convBranch1, convBranch2]
				else:
					convBranch = self.build_branch("convBranch", input_shape, number_convolutional_layers, filters_size, strides,
				                           filters_numbers, weight_decay, ndense_units, dropout, lstm, lstm_number, dropout_lstm, L2_norm_lstm, activation_fun)

		if lstm == 0 or lstm == 5 or lstm == 6 or lstm == 7:
			main_branch = convBranch(lstm_layer)
		elif lstm >= 8 and lstm <= 15:
			output1 = convBranch[0](lstm_layer)
			output2 = convBranch[1](input)

			if lstm == 10 or lstm == 15:
				# Add max layer
				output1 = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="embedL2_1")(output1)
				output2 = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="embedL2_2")(output2)

			# Add concat layer
			main_branch = tf.keras.layers.Concatenate(axis=1)([output1, output2])

			# # Add dense layer + dropout
			# main_branch = tf.keras.layers.Dense(1024, name="dense")(main_branch)
			# if dropout > 0:
			# 	main_branch = tf.keras.layers.Dropout(dropout, name="drop")(main_branch)
			# output = tf.keras.layers.Dense(512, name="code")(main_branch)
			# if dropout > 0:
			# 	output = tf.keras.layers.Dropout(dropout, name="drop_code")(output)

			if lstm == 11 or lstm == 12 or lstm == 15:
				main_branch = tf.keras.layers.Dropout(dropout, name="drop_concat")(main_branch)

			if lstm != 9 and lstm != 12 and lstm != 15:
				# Add dense layer + dropout
				main_branch = tf.keras.layers.Dense(ndense_units * 2, name="dense")(main_branch)
				if dropout > 0:
					main_branch = tf.keras.layers.Dropout(dropout, name="drop")(main_branch)
#			output = tf.keras.layers.Dense(ndense_units, name="code")(main_branch)

		else:
			if from3dto2d:
				main_branch = convBranch
			else:
				main_branch = convBranch(input)

		output = tf.keras.layers.Dense(ndense_units, name="code")(main_branch)

		if loss_mode == 'crossentropy':
			# Insert a classification layer here
			main_branch = tf.keras.layers.Dense(nclasses, activation='softmax', kernel_initializer='he_uniform', name="probs")(output)
			outputs.append(main_branch)
			losses.append('sparse_categorical_crossentropy')
			metrics.append(tf.keras.metrics.SparseCategoricalAccuracy())
		elif loss_mode == 'triplet':
			main_branch = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(output)
			outputs.append(main_branch)
			losses.append(tfa.losses.TripletSemiHardLoss(margin=margin))
			metrics = None
		elif loss_mode == 'triplet_crossentropy':
			main_branch = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(output)
			outputs.append(main_branch)
			losses.append(tfa.losses.TripletSemiHardLoss(margin=margin))
			main_branch_ = tf.keras.layers.Dense(nclasses, activation='softmax', kernel_initializer='he_uniform', name="probs")(output)
			outputs.append(main_branch_)
			losses.append('sparse_categorical_crossentropy')
			metrics = None
		elif loss_mode == 'tripletall':
#			main_branch = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(output)
#			outputs.append(main_branch)
			outputs.append(output)
			losses.append(TripletBatchAllLoss(margin=margin))
			metrics = None
		elif loss_mode == 'tripletallcross':
			main_branch1 = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(output)
			outputs.append(main_branch1)
			losses.append(TripletBatchAllLoss(margin=margin))
			main_branch = tf.keras.layers.Dense(nclasses, activation='softmax', kernel_initializer='he_uniform', name="probs")(output)
			outputs.append(main_branch)
			losses.append('sparse_categorical_crossentropy')
			metrics = {'probs': tf.keras.metrics.SparseCategoricalAccuracy()}
		elif loss_mode == 'tripletboth':
			main_branch = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(output)
			outputs.append(main_branch)
			losses.append(tfa.losses.TripletSemiHardLoss(margin=margin))
			outputs.append(main_branch)
			losses.append(tfa.losses.TripletHardLoss(margin=margin))
			metrics = None
		elif loss_mode == 'triplet_drop':
			output = tf.keras.layers.Dropout(dropout, name="drop_code")(output)
			main_branch = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(output)
			outputs.append(main_branch)
			losses.append(tfa.losses.TripletSemiHardLoss(margin=margin))
			metrics = None
		elif loss_mode == 'crosstriplet':
#			main_branch = output
			main_branch = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(output)
			outputs.append(main_branch)
			losses.append(TripletSemiHardCrossentropyLoss(margin=margin))
			metrics = None
		elif loss_mode == 'crosstriplet_nol2':
			main_branch = output
			outputs.append(main_branch)
			losses.append(TripletSemiHardCrossentropyLoss(margin=margin))
			metrics = None
		elif loss_mode == 'both_crosstriplet':
			main_branch_ = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(output)
			outputs.append(main_branch_)
			main_branch = tf.keras.layers.Dense(nclasses, activation='softmax', kernel_initializer='he_uniform', name="probs")(output)
			outputs.append(main_branch)
			losses.append(TripletSemiHardCrossentropyLoss(margin=margin))
			losses.append('sparse_categorical_crossentropy')
			metrics = None
		elif loss_mode == 'both_batchall':
			main_branch_ = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(output)
			outputs.append(main_branch_)
			main_branch = tf.keras.layers.Dense(nclasses, activation='softmax', kernel_initializer='he_uniform', name="probs")(output)
			outputs.append(main_branch)
			losses.append(TripletBatchAllCrossentropyLoss(margin=margin))
			losses.append('sparse_categorical_crossentropy')
			metrics = None
		elif loss_mode == 'both_batchall_drop':
			output = tf.keras.layers.Dropout(dropout, name="drop_code")(output)
			main_branch_ = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(output)
			outputs.append(main_branch_)
			main_branch = tf.keras.layers.Dense(nclasses, activation='softmax', kernel_initializer='he_uniform', name="probs")(output)
			outputs.append(main_branch)
			losses.append(TripletBatchAllCrossentropyLoss(margin=margin))
			losses.append('sparse_categorical_crossentropy')
			metrics = None
		elif loss_mode == 'both_hardcros':
			main_branch_ = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(output)
			outputs.append(main_branch_)
			main_branch = tf.keras.layers.Dense(nclasses, activation='softmax', kernel_initializer='he_uniform', name="probs")(output)
			outputs.append(main_branch)
			losses.append(TripletHardCrossentropyLoss(margin=margin))
			losses.append('sparse_categorical_crossentropy')
			metrics = None
		elif loss_mode == 'both_batchalldist':
			main_branch_ = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="signature")(output)
			outputs.append(main_branch_)
			main_branch = tf.keras.layers.Dense(nclasses, activation='softmax', kernel_initializer='he_uniform', name="probs")(output)
			outputs.append(main_branch)
			losses.append(TripletBatchAllLoss(margin=margin))
			losses.append('sparse_categorical_crossentropy')
			metrics = None
		else:
			import sys
			sys.exit('Wrong losses')

		self.model = tf.keras.Model(inputs=input, outputs=outputs)
		self.model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)

		# Save useful info for recovering the model with different Python versions
		modelpars = {'filters_size': filters_size,
		             'filters_numbers': filters_numbers,
		             'input_shape': input_shape,
		             'ndense_units': ndense_units,
		             'weight_decay': weight_decay,
		             'dropout': dropout,
		             #'optimizer': optimizer,
		             'custom': 'TripletSemiHardLoss',
		             'nclasses': nclasses,
		             'use3D': use3D,
		             'model_version': model_version,
		             'loss_mode': loss_mode,
		             'loss_weights': loss_weights,
		             'margin': margin}
		dd.io.save(os.path.join(self.experdir, "model-config.hdf5"), modelpars)

	def build_branch(self, name, input_shape=(50, 60, 60), number_convolutional_layers=4, filters_size=None,
	                 strides=None, filters_numbers=None, weight_decay=0.0005, ndense_units=4096, dropout=0.4, lstm=-1,
					 lstm_number=512, dropout_lstm=0.0, L2_norm_lstm=None, add_dense=True, activation_fun='relu'):
		if filters_numbers is None:
			filters_numbers = [96, 192, 512, 4096]
		L2_norm = tf.keras.regularizers.l2(weight_decay)
		if L2_norm_lstm is not None:
			L2_norm_lstm = tf.keras.regularizers.l2(L2_norm_lstm)

		convBranch = tf.keras.Sequential(name=name)
		if lstm == 2:
			if activation_fun == 'leaky':
				convBranch.add(tf.keras.layers.TimeDistributed(
					tf.keras.layers.Conv2D(filters_numbers[0], kernel_size=(filters_size[0]), strides=(strides[0]),
					                       kernel_regularizer=L2_norm, activation=None, input_shape=input_shape,
					                       data_format='channels_first')))
				convBranch.add(tf.keras.layers.TimeDistributed(tf.keras.layers.LeakyReLU()))
			else:
				convBranch.add(tf.keras.layers.TimeDistributed(
					tf.keras.layers.Conv2D(filters_numbers[0], kernel_size=(filters_size[0]), strides=(strides[0]),
					                       kernel_regularizer=L2_norm, activation='relu', input_shape=input_shape,
					                       data_format='channels_first')))

			convBranch.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_first')))

			for i in range(1, number_convolutional_layers):
				if activation_fun == 'leaky':
					convBranch.add(tf.keras.layers.TimeDistributed(
						tf.keras.layers.Conv2D(filters_numbers[i], kernel_size=(filters_size[i]), strides=(strides[i]),
						                       kernel_regularizer=L2_norm, activation=None,
						                       data_format='channels_first')))
					convBranch.add(tf.keras.layers.TimeDistributed(tf.keras.layers.LeakyReLU()))
				else:
					convBranch.add(tf.keras.layers.TimeDistributed(
						tf.keras.layers.Conv2D(filters_numbers[i], kernel_size=(filters_size[i]), strides=(strides[i]),
						                       kernel_regularizer=L2_norm, activation='relu',
						                       data_format='channels_first')))

				#if i != number_convolutional_layers - 1:
				convBranch.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_first')))

			convBranch.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(name="flatten")))
			convBranch.add(tf.keras.layers.LSTM(lstm_number, dropout=dropout_lstm, kernel_regularizer=L2_norm_lstm))

			if add_dense:
				# Add dense layer + dropout
				convBranch.add(tf.keras.layers.Dense(ndense_units, name="dense"))
				if dropout > 0:
					convBranch.add(tf.keras.layers.Dropout(dropout, name="drop"))
		else:
			if lstm == 1:
				convBranch.add(tf.keras.layers.ConvLSTM2D(filters_numbers[0], kernel_size=(filters_size[0]), strides=(strides[0]),
					                       kernel_regularizer=L2_norm, input_shape=input_shape,
					                       data_format='channels_first'))
			else:
				if activation_fun == 'leaky':
					convBranch.add(tf.keras.layers.Conv2D(filters_numbers[0], kernel_size=(filters_size[0]), strides=(strides[0]),
						                       kernel_regularizer=L2_norm, activation=None,
						                       input_shape=input_shape,
						                       data_format='channels_first'))
					convBranch.add(tf.keras.layers.TimeDistributed(tf.keras.layers.LeakyReLU()))
				else:
					convBranch.add(tf.keras.layers.Conv2D(filters_numbers[0], kernel_size=(filters_size[0]), strides=(strides[0]),
				                                    kernel_regularizer=L2_norm, activation='relu', input_shape=input_shape,
				                                    data_format='channels_first')) #

			convBranch.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))#

			for i in range(1, number_convolutional_layers):
				if activation_fun == 'leaky':
					convBranch.add(tf.keras.layers.Conv2D(filters_numbers[i], kernel_size=(filters_size[i]), strides=(strides[i]),
						                       kernel_regularizer=L2_norm, activation=None,
						                       data_format='channels_first'))
					convBranch.add(tf.keras.layers.TimeDistributed(tf.keras.layers.LeakyReLU()))
				else:
					convBranch.add(tf.keras.layers.Conv2D(filters_numbers[i], kernel_size=(filters_size[i]), strides=(strides[i]),
					                                    kernel_regularizer=L2_norm, activation='relu',
					                                    data_format='channels_first')) #

				#if i != number_convolutional_layers - 1:
				convBranch.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))

			convBranch.add(tf.keras.layers.Flatten(name="flatten"))

			if add_dense:
				# Add dense layer + dropout
				convBranch.add(tf.keras.layers.Dense(ndense_units, name="dense"))
				if dropout > 0:
					convBranch.add(tf.keras.layers.Dropout(dropout, name="drop"))

		return convBranch

	def build_branch_Manuel(self, name, input_shape=(50, 60, 60), number_convolutional_layers=4, filters_size=None,
	                 strides=None, filters_numbers=None, weight_decay=0.0005, ndense_units=1024, dropout=0.4, lstm=-1,
							lstm_number=512, dropout_lstm=0.0, L2_norm_lstm=None, add_dense=True, activation_fun='relu',
							pruning=False, frequency=100, begin_step=2000, target_sparsity=0.9):
		if filters_numbers is None:
			filters_numbers = [96, 192, 512, 512]
		L2_norm = tf.keras.regularizers.l2(weight_decay)
		if L2_norm_lstm is not None:
			L2_norm_lstm = tf.keras.regularizers.l2(L2_norm_lstm)

		convBranch = tf.keras.Sequential(name=name)

		if lstm == 2:
			if activation_fun == 'leaky':
				convBranch.add(tf.keras.layers.TimeDistributed(
					tf.keras.layers.Conv2D(filters_numbers[0], kernel_size=(filters_size[0]), strides=(strides[0]),
					                       kernel_regularizer=L2_norm, activation=None, input_shape=input_shape,
					                       data_format='channels_first')))
				convBranch.add(tf.keras.layers.TimeDistributed(tf.keras.layers.LeakyReLU()))
			else:
				convBranch.add(tf.keras.layers.TimeDistributed(
					tf.keras.layers.Conv2D(filters_numbers[0], kernel_size=(filters_size[0]), strides=(strides[0]),
					                       kernel_regularizer=L2_norm, activation='relu',
					                       input_shape=input_shape,
					                       data_format='channels_first')))

			convBranch.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_first')))

			for i in range(1, number_convolutional_layers):
				if activation_fun == 'leaky':
					convBranch.add(tf.keras.layers.TimeDistributed(
						tf.keras.layers.Conv2D(filters_numbers[i], kernel_size=(filters_size[i]), strides=(strides[i]),
						                       kernel_regularizer=L2_norm, activation=None,
						                       data_format='channels_first')))
					convBranch.add(tf.keras.layers.TimeDistributed(tf.keras.layers.LeakyReLU()))
				else:
					convBranch.add(tf.keras.layers.TimeDistributed(
						tf.keras.layers.Conv2D(filters_numbers[i], kernel_size=(filters_size[i]), strides=(strides[i]),
						                       kernel_regularizer=L2_norm, activation='relu',
						                       data_format='channels_first')))

				#if i != number_convolutional_layers - 1:
				convBranch.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_first')))

			convBranch.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(name="flatten")))
			convBranch.add(tf.keras.layers.LSTM(lstm_number, dropout=dropout_lstm, kernel_regularizer=L2_norm_lstm))

			if add_dense:
				# Add dense layer + dropout
				convBranch.add(tf.keras.layers.Dense(ndense_units * 2, name="dense"))
				if dropout > 0:
					convBranch.add(tf.keras.layers.Dropout(dropout, name="drop"))
				convBranch.add(tf.keras.layers.Dense(ndense_units, name="code"))
		else:
			if lstm == 1:
				convBranch.add(tf.keras.layers.ConvLSTM2D(filters_numbers[0], kernel_size=(filters_size[0]), strides=(strides[0]),
					                       kernel_regularizer=L2_norm, input_shape=input_shape,
					                       data_format='channels_first'))
			else:
				if activation_fun == 'leaky':
					convBranch.add(
						tf.keras.layers.Conv2D(filters_numbers[0], kernel_size=(filters_size[0]), strides=(strides[0]),
						                       kernel_regularizer=L2_norm, activation=None,
						                       input_shape=input_shape,
						                       data_format='channels_first'))
					convBranch.add(tf.keras.layers.TimeDistributed(tf.keras.layers.LeakyReLU()))
				else:
					convBranch.add(tf.keras.layers.Conv2D(filters_numbers[0], kernel_size=(filters_size[0]), strides=(strides[0]),
				                                    kernel_regularizer=L2_norm, activation='relu', input_shape=input_shape,
				                                    data_format='channels_first'))

			convBranch.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))

			for i in range(1, number_convolutional_layers):
				if activation_fun == 'leaky':
					convBranch.add(tf.keras.layers.Conv2D(filters_numbers[i], kernel_size=(filters_size[i]), strides=(strides[i]),
					                                    kernel_regularizer=L2_norm, activation=None,
					                                    data_format='channels_first'))
					convBranch.add(tf.keras.layers.TimeDistributed(tf.keras.layers.LeakyReLU()))
				else:
					convBranch.add(
						tf.keras.layers.Conv2D(filters_numbers[i], kernel_size=(filters_size[i]), strides=(strides[i]),
						                       kernel_regularizer=L2_norm, activation='relu',
						                       data_format='channels_first'))

				#if i != number_convolutional_layers - 1:
				convBranch.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))

			convBranch.add(tf.keras.layers.Flatten(name="flatten"))

			if add_dense:
				# Add dense layer + dropout
				convBranch.add(tf.keras.layers.Dense(ndense_units * 2, name="dense"))
				if dropout > 0:
					convBranch.add(tf.keras.layers.Dropout(dropout, name="drop"))
				convBranch.add(tf.keras.layers.Dense(ndense_units, name="code"))
		# Pruning
		if pruning:
			print("He llegado a pruning")
			pruning_params = {
				"pruning_schedule": pruning_schedule.ConstantSparsity(target_sparsity, begin_step=begin_step, frequency=frequency)}
			convBranch = prune.prune_low_magnitude(convBranch, **pruning_params)

		return convBranch

	def build_3Dbranch(self, name, input_shape=(50, 60, 60), number_convolutional_layers=4, filters_size=None,
	                 strides=None, filters_numbers=None, weight_decay=0.0005, ndense_units=4096, dropout=0.4):
		if filters_numbers is None:
			filters_numbers = [96, 192, 512, 4096]
		L2_norm = tf.keras.regularizers.l2(weight_decay)

		convBranch = tf.keras.Sequential(name=name)

		convBranch.add(tf.keras.layers.Conv3D(filters_numbers[0], kernel_size=(3, filters_size[0], filters_size[0]),
		                                      strides=(1, strides[0], strides[0]), kernel_regularizer=L2_norm,
		                                      activation='relu', input_shape=input_shape, data_format='channels_last'))

		convBranch.add(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), data_format='channels_last'))

		for i in range(1, number_convolutional_layers):
			convBranch.add(tf.keras.layers.Conv2D(filters_numbers[i], kernel_size=(3, filters_size[i], filters_size[i]),
			                                      strides=(1, strides[i], strides[i]), kernel_regularizer=L2_norm,
			                                      activation='relu', data_format='channels_last'))

			if i != number_convolutional_layers - 1:
				convBranch.add( tf.keras.layers.Conv2D(filters_numbers[i], kernel_size=(3, filters_size[i], filters_size[i]),
					                       strides=(1, strides[i], strides[i]), kernel_regularizer=L2_norm,
					                       activation='relu', data_format='channels_last'))

				convBranch.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2, 2), data_format='channels_last'))
			else:
				convBranch.add(tf.keras.layers.Conv2D(filters_numbers[i], kernel_size=(filters_size[i]),
					                       strides=(1, strides[i], strides[i]), kernel_regularizer=L2_norm,
					                       activation='relu', data_format='channels_last'))

		convBranch.add(tf.keras.layers.Flatten(name="flatten"))

		# Add dense layer + dropout
		convBranch.add(tf.keras.layers.Dense(ndense_units, name="dense"))
		if dropout > 0:
			convBranch.add(tf.keras.layers.Dropout(dropout, name="drop"))

		return convBranch

	def build_3Dbranch_Manuel(self, name, input_shape=(25, 60, 60, 1), ndense_units=512, pruning=False,
							frequency = 100, begin_step = 2000,target_sparsity = 0.9):
		convBranch = tf.keras.Sequential(name=name)

		convBranch.add(tf.keras.layers.Conv3D(64, (3, 5, 5), strides=(1, 2, 2), padding='valid', activation='relu',
		                        input_shape=input_shape, data_format='channels_last'))

		convBranch.add(tf.keras.layers.Conv3D(128, (3, 3, 3), strides=(1, 2, 2), padding='valid', activation='relu',
		                        input_shape=input_shape, data_format='channels_last'))

		convBranch.add(tf.keras.layers.Conv3D(256, (3, 3, 3), strides=(2, 2, 2), padding='valid', activation='relu',
		                        input_shape=input_shape, data_format='channels_last'))

		convBranch.add(tf.keras.layers.Conv3D(512, (3, 3, 3), strides=(2, 2, 2), padding='valid', activation='relu',
		                        input_shape=input_shape, data_format='channels_last'))

		convBranch.add(tf.keras.layers.Conv3D(512, (3, 2, 2), strides=(1, 1, 1), padding='valid', activation='relu',
		                        input_shape=input_shape, data_format='channels_last'))

		convBranch.add(tf.keras.layers.Conv3D(512, (2, 1, 1), strides=(1, 1, 1), padding='valid', activation='relu',
		                        input_shape=input_shape, data_format='channels_last'))

		# Dense without activation function
		convBranch.add(tf.keras.layers.Conv3D(ndense_units, (1, 1, 1), strides=(1, 1, 1), activation=None,
		                        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
		                        kernel_initializer='he_uniform', name="grayCode"))

		convBranch.add(tf.keras.layers.Flatten())

		# Pruning
		if pruning:
			print("He llegado a pruning")
			pruning_params = {
				"pruning_schedule": pruning_schedule.ConstantSparsity(target_sparsity, begin_step=begin_step, frequency=frequency)}
			convBranch = prune.prune_low_magnitude(convBranch, **pruning_params)


		return convBranch

	def branch3dto2d1d(self, name, batch_size, input, input_shape=(25, 60, 60, 1), ndense_units=512, pruning=False,
							frequency = 100, begin_step = 2000,target_sparsity = 0.9):

		model = tf.reshape(input, [-1, input_shape[1], input_shape[2], input_shape[3]])
		model = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='valid', activation='relu',
		                        input_shape=input_shape[1:], data_format='channels_last')(model)
		shape_ = model.get_shape().as_list()
		model = tf.reshape(model, [batch_size, -1, shape_[1] * shape_[2], shape_[3]])
		model = tf.keras.layers.Conv2D(64, (3, 1), strides=1, padding='valid', activation='relu',
									   data_format='channels_last')(model)
		model = tf.reshape(model, [-1, shape_[1], shape_[2], shape_[3]])
		#conv2
		model = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='valid', activation='relu',
									   data_format='channels_last')(model)
		shape_ = model.get_shape().as_list()
		model = tf.reshape(model, [batch_size, -1, shape_[1] * shape_[2], shape_[3]])
		model = tf.keras.layers.Conv2D(128, (3, 1), strides=1, padding='valid', activation='relu',
									   data_format='channels_last')(model)
		model = tf.reshape(model, [-1, shape_[1], shape_[2], shape_[3]])

		#conv3
		model = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='valid', activation='relu',
									   data_format='channels_last')(model)
		shape_ = model.get_shape().as_list()
		model = tf.reshape(model, [batch_size, -1, shape_[1] * shape_[2], shape_[3]])
		model = tf.keras.layers.Conv2D(256, (3, 1), strides=(2,1), padding='valid', activation='relu',
									   data_format='channels_last')(model)
		model = tf.reshape(model, [-1, shape_[1], shape_[2], shape_[3]])

		#conv4
		model = tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='valid', activation='relu',
									   data_format='channels_last')(model)
		shape_ = model.get_shape().as_list()
		model = tf.reshape(model, [batch_size, -1, shape_[1] * shape_[2], shape_[3]])
		model = tf.keras.layers.Conv2D(512, (3, 1), strides=(2, 1), padding='valid', activation='relu',
									   data_format='channels_last')(model)
		model = tf.reshape(model, [-1, shape_[1], shape_[2], shape_[3]])

		#conv5
		model = tf.keras.layers.Conv2D(512, (2, 2), strides=(1, 1), padding='valid', activation='relu',
									   data_format='channels_last')(model)
		shape_ = model.get_shape().as_list()
		model = tf.reshape(model, [batch_size, -1, shape_[1] * shape_[2], shape_[3]])
		model = tf.keras.layers.Conv2D(512, (3, 1), strides=(1, 1), padding='valid', activation='relu',
									   data_format='channels_last')(model)
		model = tf.reshape(model, [-1, shape_[1], shape_[2], shape_[3]])

		#conv6
		model = tf.keras.layers.Conv2D(512, (1, 1), strides=(1, 1), padding='valid', activation='relu',
									   data_format='channels_last')(model)
		shape_ = model.get_shape().as_list()
		model = tf.reshape(model, [batch_size, -1, shape_[1] * shape_[2], shape_[3]])
		model = tf.keras.layers.Conv2D(512, (2, 1), strides=(1, 1), padding='valid', activation='relu',
									   data_format='channels_last')(model)

		model = tf.reshape(model, [batch_size,512])

		model = tf.keras.layers.Dense(ndense_units, activation=None,
		                        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
		                        kernel_initializer='he_uniform', name="grayCode")(model)


		model = tf.keras.layers.Dropout(0.4, name="drop")(model)

		# Pruning
		if pruning:
			print("He llegado a pruning")
			pruning_params = {
				"pruning_schedule": pruning_schedule.ConstantSparsity(target_sparsity, begin_step=begin_step, frequency=frequency)}
			convBranch = prune.prune_low_magnitude(convBranch, **pruning_params)


		return model

	def fit(self, epochs, callbacks, training_generator, validation_generator, current_step=0, validation_steps=None, encode_layer=None, steps_per_epoch=None):
		self.hist = self.model.fit(training_generator, validation_data=validation_generator, epochs=epochs,
		                           callbacks=callbacks, validation_steps=validation_steps, initial_epoch=current_step,
		                           verbose=1, steps_per_epoch=steps_per_epoch) #, workers=4, max_queue_size=10, use_multiprocessing=True)

		if encode_layer is None:
			self.model_encode = tf.keras.Model(self.model.input, self.model.layers[-1].input)
		else:
			out_layer = self.model.get_layer(encode_layer).output
			self.model_encode = tf.keras.Model(self.model.input, out_layer)
		return len(self.hist.epoch)

	def predict(self, data, batch_size=128):
		pred = self.model.predict(data, batch_size=batch_size)
		return pred

	def encode(self, data):
		features = self.model_encode(data)

		# L2 normalize embeddings
		codes_norm_tf = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(features)

		# Get the numpy matrix
		codes_norm = codes_norm_tf.numpy()
		#codes_norm = features.numpy()
		return codes_norm

	def save(self, epoch=None):
		if epoch is not None:
			self.model.save(os.path.join(self.experdir, "model-state-{:04d}.hdf5".format(epoch)))

			# Save in such a way that can be recovered from different Python versions
			self.model.save_weights(os.path.join(self.experdir, "model-state-{:04d}_weights.hdf5".format(epoch)))
		else:
			self.model.save(os.path.join(self.experdir, "model-final.hdf5"))

			# Save in such a way that can be recovered from different Python versions
			self.model.save_weights(os.path.join(self.experdir, "model-final_weights.hdf5"))

