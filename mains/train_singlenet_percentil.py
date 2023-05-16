import sys
import os
from os.path import expanduser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import os.path as osp
import pathlib

maindir = pathlib.Path(__file__).parent.absolute()
if sys.version_info[1] >= 6:
	sys.path.insert(0, osp.join(maindir, ".."))
else:
	sys.path.insert(0, str(maindir) + "/..")
# --------------------------------
homedir = expanduser("~")

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

from tensorflow.keras import optimizers
import deepdish as dd
from nets.single_model import SingleGaitModel
from nets.single_model_desenrollado import SingleGaitModelDesen
from tensorflow.keras.callbacks import ReduceLROnPlateau

from data.dataGenerator import DataGeneratorGait
from utils.mj_netUtils import mj_findLatestFileModel
from callbacks.model_saver import ModelSaver
from tensorboard.plugins import projector
from callbacks.counts_zero import countszeros
from kerassurgeon.operations import delete_channels

import tensorflow_model_optimization as tfmot

# ===============================================================
def trainGaitNet(datadir="matimdbtum_gaid_N150_of25_60x60_lite", experfix="of",
                 nclasses=0, lr=0.01, dropout=0.4,
                 experdirbase=".", epochs=15, batchsize=150, optimizer="SGD",
                 modality="of", initnet="", use3D=False, freeze_all=False, nofreeze=False,
                 logdir="", extra_epochs=0, model_version='IWANN', cameras=None, sufix=None, verbose=0,
				 pruning=False, frequency=100, begin_step=2000, target_sparsity=0.9, pstruc=False):
	"""
	Trains a CNN for gait recognition
	:param datadir: root dir containing dd files
	:param experfix: string to customize experiment name
	:param nclasses: number of classes
	:param lr: starting learning rate
	:param dropout: dropout value
	:param tdim: extra dimension of samples. Usually 50 for OF, and 25 for gray and depth
	:param epochs: max number of epochs
	:param batchsize: integer
	:param optimizer: options like {SGD, Adam,...}
	:param logdir: path to save Tensorboard info
	:param ndense_units: number of dense units for last FC layer
	:param verbose: integer
	:return: model, experdir, accuracy
	"""

	if use3D:
		if modality == 'of':
			input_shape = (25, 60, 60, 2)
		elif modality == 'rgb':
			input_shape = (25, 60, 60, 3)
		else:
			input_shape = (25, 60, 60, 1)
	else:
		if modality == 'of':
			input_shape = (50, 60, 60)
		elif modality == 'rgb':
			input_shape = (75, 60, 60)
		else:
			input_shape = (25, 60, 60)

	if model_version == 'iwann':
		number_convolutional_layers = 4
		filters_size = [(7, 7), (5, 5), (3, 3), (2, 2)]
		filters_numbers = [96, 192, 512, 4096]
		ndense_units = 2048
		strides = [1, 2, 1, 1]
	elif model_version == 'bmvc':
		number_convolutional_layers = 4
		filters_size = [(7, 7), (5, 5), (3, 3), (2, 2)]
		filters_numbers = [96, 192, 512, 512]
		if nclasses == 74 or nclasses == 50:
			ndense_units = 2048
		else:
			ndense_units = 1024
		strides = [1, 1, 1, 1]
	else:
		number_convolutional_layers = 4
		filters_size = [(7, 7), (5, 5), (3, 3), (2, 2)]
		filters_numbers = [96, 192, 512, 4096]
		ndense_units = 2048
		strides = [1, 2, 1, 1]

	weight_decay = 0.00005
	momentum = 0.9

	optimfun = optimizers.Adam(lr=lr)
	infix = "_opAdam"
	if optimizer != "Adam":
		infix = "_op" + optimizer
		if optimizer == "SGD":
			optimfun = optimizers.SGD(lr=lr, momentum=momentum)
		elif optimizer == "AMSGrad":
			optimfun = optimizers.Adam(lr=lr, amsgrad=True)
		else:
			optimfun = eval("optimizers." + optimizer + "(lr=initialLR)")

	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
	                              patience=3, min_lr=0.00001)
	es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1)

	if use3D:
		infix = "_" + modality + "3D" + infix
	else:
		infix = "_" + modality + infix

	if nofreeze:
		freeze_convs = False
	else:
		freeze_convs = True
	if initnet != "" and freeze_all:
		infix = infix + "_frall"

	if pruning:
		infix = infix + '_fq{:3d}_begins{:4d}_ts{:0.2f}'.format(frequency, begin_step, target_sparsity)

	# Create a TensorBoard instance with the path to the logs directory
	subdir = experfix + '_' + model_version + '_N{:03d}_datagen{}_bs{:03d}_lr{:0.6f}_dr{:0.2f}'.format(nclasses, infix, batchsize, lr,
	                                                                             dropout)  # To be customized
	if sufix is not None:
		subdir = subdir + "_" + sufix
	experdir = osp.join(experdirbase, subdir)
	if verbose > 0:
		print(experdir)
	if not osp.exists(experdir):
		os.makedirs(experdir)

	#################Structural Pruning#################
	####NOTES: Search #TODO
	# 1. Take into account the percentile to remove filters (1st todo).
	# 2. If 2D model, change the name of last dense layer before break the sequence (2nd #todo (code_1)).
	# 3. If model with 150 classes, remove last dense layer for KNN. If 155 classes, let last dense layer.

	if pstruc:
		model = SingleGaitModel(experdir)
		if os.path.exists(os.path.join(experdir, 'model-without-last-layer.hdf5')):
			previous_model = os.path.join(experdir, 'model-without-last-layer.hdf5')
		else:
			if os.path.exists(os.path.join(experdir, 'model-final.hdf5')):
				previous_model = os.path.join(experdir, 'model-final.hdf5')
			else:
				pattern_file = "model-state-{:04d}.hdf5"
				previous_model = mj_findLatestFileModel(experdir, pattern_file, epoch_max=4000)

		print(previous_model)
		model.load(previous_model)
		model.model.summary()

		print("New model will be created")
		initnet = ""
		experdir_estruc = '/home/pruiz/experiments_gait_multimodal/pestruc'
		model_new = SingleGaitModelDesen(experdir_estruc)
		model_new.build_or_load(input_shape, number_convolutional_layers, filters_size, filters_numbers, strides,
							ndense_units, weight_decay, dropout, optimizer=optimfun, nclasses=nclasses,
							initnet=initnet, freeze_convs=freeze_convs, use3D=use3D, freeze_all=freeze_all,
							model_version=model_version, pruning=pruning, frequency=frequency,
							begin_step=begin_step, target_sparsity=target_sparsity)


		# If 2D model, change the name of last dense layer before break the sequence
		# In single model change line 374, dense name --> code_1 #TODO
		#model.model.layers[2]._init_set_name("code_1") #TODO

		model_new.model.summary()


		for i in range(len(model_new.model.layers)):
			for j in range(len(model.model.layers)):
				if 'Sequential' in model.model.layers[j]._keras_api_names[0]:
					for k in range(len(model.model.layers[j].layers)):
						if model.model.layers[j].layers[k].name == model_new.model.layers[i].name:
							model_new.model.layers[i].set_weights(model.model.layers[j].layers[k].get_weights())
				if model_new.model.layers[i].name == model.model.layers[j].name:
					model_new.model.layers[i].set_weights(model.model.layers[j].get_weights())

		#path_save = experdir + '/model_unrolled.h5'
		#model_new.model.save(path_save)
		#print("Unrolled model stored")

		w_list = []
		layer_name = []
		values_total = []

		for i in range(len(model_new.model.layers)): # Sequential model
			if ('keras.layers.Conv3D' in model_new.model.layers[i]._keras_api_names[0] or 'keras.layers.Conv2D' in model_new.model.layers[i]._keras_api_names[0]) and model_new.model.layers[i].name != 'grayCode':  # take convolutional layers
				w = model_new.model.layers[i].get_weights()[0]
				w_list.append(w)
				layer_name.append(model_new.model.layers[i].name)

		for i in range(len(w_list)):
			weight = w_list[i]
			weight_dict = {}
			if not use3D:
				num_filters = len(weight[0, 0, 0, :])
			else:
				num_filters = len(weight[0, 0, 0, 0, :])

			# L1-norm of every filter and save in a dictionary
			for j in range(num_filters):
				if not use3D:
					w_s = np.sum(abs(weight[:, :, :, j]))

				else:
					w_s = np.sum(abs(weight[:, :, :, :, j]))
				weight_dict[j] = w_s

			values = np.fromiter(weight_dict.values(), dtype=float)
			values_total= np.append(values_total, values)

			# Filters in ascending L1 value
			weights_dict_sort = sorted(weight_dict.items(), key=lambda k: k[1])
			print("L1 norm conv layer {}\n".format(i + 1), weights_dict_sort)

		values_total = np.sort(values_total)
		value_percentile = 50#TODO
		percentile = np.percentile(values_total, value_percentile)

		contador = 0
		deleted = 0
		model_filters = model_new.model


		for i in range(len(model_filters.layers)):  # Sequential model
			if model_filters.layers[i].name in layer_name:
				weight = w_list[contador]

				if not use3D:
					num_filters = len(weight[0, 0, 0, :])
				else:
					num_filters = len(weight[0, 0, 0, 0, :])

				# L1-norm of every filter and save in a dictionary
				for j in reversed(range(num_filters)):
					if not use3D:
						w_s = np.sum(abs(weight[:, :, :, j]))

					else:
						w_s = np.sum(abs(weight[:, :, :, :, j]))

					if  not use3D and w_s < percentile and model_filters.layers[i].output_shape[1] > 1:
						model_filters = delete_channels(model_filters, model_filters.layers[i], np.array([j]))
						deleted = deleted + 1
					if use3D and w_s < percentile and model_filters.layers[i].output_shape[-1] > 1:
						model_filters = delete_channels(model_filters, model_filters.layers[i], np.array([j]))
						deleted = deleted + 1

				contador = contador + 1


		print("{} Filters removed".format(deleted))
		#model_filters.summary()

		path_save = experdir + '/model_without_{}_filters_{}_{:.2f}.h5'.format(deleted, value_percentile, percentile)
		model_filters.save(path_save)
		print("Model with {} less filters saved".format(deleted))
		print("END")

	else:
		# Prepare model
		pattern_file = "model-state-{:04d}.hdf5"
		previous_model = mj_findLatestFileModel(experdir, pattern_file, epoch_max=epochs)
		print(previous_model)
		if os.path.exists(os.path.join(experdir, 'model-final.hdf5')):
			print("Already trained model, skipping.")
			return None, None
		else:
			model = SingleGaitModel(experdir)
			if previous_model != "":
				pms = previous_model.split("-")
				initepoch = int(pms[len(pms) - 1].split(".")[0])
				print("* Info: a previous model was found. Warming up from it...[{:d}]".format(initepoch))
				from tensorflow.keras.models import load_model
				model.load(previous_model)

			else:
				if initnet != "":
					print("* Model will be init from: " + initnet)

				model.build_or_load(input_shape, number_convolutional_layers, filters_size, filters_numbers, strides,
									ndense_units, weight_decay, dropout, optimizer=optimfun, nclasses=nclasses,
									initnet=initnet, freeze_convs=freeze_convs, use3D=use3D, freeze_all=freeze_all,
									model_version=model_version, pruning=pruning, frequency=frequency,
									begin_step=begin_step, target_sparsity=target_sparsity)

			model.model.summary()
			model.model.layers[1].summary()

			# Tensorboard
			if logdir == "":
				logdir = experdir
				model_saver = ModelSaver(model)

				from tensorflow.keras.callbacks import TensorBoard

				tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=False,
										  profile_batch=0)
				callbacks = [reduce_lr, tensorboard, model_saver, es_callback]
			else:  # This case is for parameter tuning
				# Save checkpoint
				model_saver = ModelSaver(model)

				from tensorflow.keras.callbacks import TensorBoard
				tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=False,
										  profile_batch=3)

				callbacks = [reduce_lr, tensorboard, model_saver]

			if pruning:
				pruning_callbacks = tfmot.sparsity.keras.UpdatePruningStep()
				counts_callbacks = countszeros(model.model)

				callbacks.append(pruning_callbacks)
				callbacks.append(counts_callbacks)
				print(callbacks)
			# ---------------------------------------
			# Prepare data
			# ---------------------------------------
			if nclasses == 150:
				data_folder = osp.join(datadir, 'tfimdb_tum_gaid_N150_train_{}25_60x60'.format(modality))
				info_file = osp.join(datadir, 'tfimdb_tum_gaid_N150_train_{}25_60x60.h5'.format(modality))
			elif nclasses == 155:
				data_folder = osp.join(datadir, 'tfimdb_tum_gaid_N155_ft_{}25_60x60'.format(modality))
				info_file = osp.join(datadir, 'tfimdb_tum_gaid_N155_ft_{}25_60x60.h5'.format(modality))
			elif nclasses == 16:
				data_folder = osp.join(datadir, 'tfimdb_tum_gaid_N016_ft_{}25_60x60'.format(modality))
				info_file = osp.join(datadir, 'tfimdb_tum_gaid_N016_ft_{}25_60x60.h5'.format(modality))
			elif nclasses == 74:
				data_folder = osp.join(datadir, 'tfimdb_casia_b_N074_train_{}25_60x60'.format(modality))
				info_file = osp.join(datadir, 'tfimdb_casia_b_N074_train_{}25_60x60.h5'.format(modality))
			elif nclasses == 50:
				data_folder = osp.join(datadir, 'tfimdb_casia_b_N050_ft_{}25_60x60'.format(modality))
				info_file = osp.join(datadir, 'tfimdb_casia_b_N050_ft_{}25_60x60.h5'.format(modality))
			else:
				sys.exit(0)

			dataset_info = dd.io.load(info_file)

			# Find label mapping for training
			if nclasses > 0:
				ulabels = np.unique(dataset_info['label'])
				# Create mapping for labels
				labmap = {}
				for ix, lab in enumerate(ulabels):
					labmap[int(lab)] = ix
			else:
				labmap = None

			# Data generators
			train_generator = DataGeneratorGait(dataset_info, batch_size=150, mode='train', labmap=labmap, modality=modality, datadir=data_folder, camera=cameras, use3D=use3D)
			val_generator = DataGeneratorGait(dataset_info, batch_size=150, mode='val', labmap=labmap, modality=modality, datadir=data_folder, camera=cameras, use3D=use3D)

			# ---------------------------------------
			# Train model
			# --------------------------------------
			if verbose > 1:
				print(experdir)
			print(callbacks)
			last_epoch = model.fit(epochs, callbacks, train_generator, val_generator)

			# Fine-tune on remaining validation samples
			if extra_epochs > 0:
				if verbose > 0:
					print("Adding validation samples to training and run for few epochs...")
				del train_generator

				train_generator = DataGeneratorGait(dataset_info, batch_size=150, mode='trainval', labmap=labmap, modality=modality,
													datadir=data_folder, camera=cameras)

				ft_epochs = last_epoch + extra_epochs  # DEVELOP!

				callbacks[0] = ReduceLROnPlateau(monitor='loss', factor=0.2,
												 patience=3, min_lr=0.00001)
				tf.keras.backend.set_value(model.model.optimizer.lr, 0.001)
				model.fit(ft_epochs, callbacks, train_generator, val_generator, last_epoch)

			# Save codes to Projector
			print("Exporting to projector")
			META_DATA_FNAME = 'meta.tsv'  # Labels will be stored here
			EMBEDDINGS_TENSOR_NAME = 'embeddings'
			os.makedirs(logdir, exist_ok=True)
			EMBEDDINGS_FPATH = os.path.join(logdir, EMBEDDINGS_TENSOR_NAME + '.ckpt')
			STEP = epochs + extra_epochs

			data = []
			labels = []
			for e in range(0, len(train_generator)):
				_X, _Y = train_generator.__getitem__(e)
				_X_codes = model.encode(_X)
				data.extend(_X_codes)
				labels.extend(_Y)

			mj_register_embedding(EMBEDDINGS_TENSOR_NAME, META_DATA_FNAME, logdir)
			mj_save_labels_tsv(labels, META_DATA_FNAME, logdir)

			tensor_embeddings = tf.Variable(data, name=EMBEDDINGS_TENSOR_NAME)
			saver = tf.compat.v1.train.Saver([tensor_embeddings])  # Must pass list or dict
			saver.save(sess=None, global_step=STEP, save_path=EMBEDDINGS_FPATH)

			return model, experdir


def mj_register_embedding(embedding_tensor_name, meta_data_fname, log_dir, sprite_path=None, image_size=(1, 1)):
	config = projector.ProjectorConfig()
	embedding = config.embeddings.add()
	embedding.tensor_name = embedding_tensor_name
	embedding.metadata_path = meta_data_fname

	if sprite_path:
		embedding.sprite.image_path = sprite_path
		embedding.sprite.single_image_dim.extend(image_size)

	projector.visualize_embeddings(log_dir, config)


def mj_save_labels_tsv(labels, filepath, log_dir):
	with open(os.path.join(log_dir, filepath), 'w') as f:
		# f.write('Class\n') # Not allowed as we have just one column
		for label in labels:
			f.write('{}\n'.format(label))


################# MAIN ################
if __name__ == "__main__":
	import argparse

	# Input arguments
	parser = argparse.ArgumentParser(description='Trains a CNN for gait')
	parser.add_argument('--debug', default=False, action='store_true')
	parser.add_argument('--use3d', default=False, action='store_true', help="Use 3D convs in 2nd branch?")
	parser.add_argument('--freezeall', default=False, action='store_true', help="Freeze all weights?")
	parser.add_argument('--nofreeze', default=False, action='store_true', help="Avoid freezing any weight?")
	parser.add_argument('--dropout', type=float, required=False,
	                    default=0.4, help='Dropout value for after-fusion layers')
	parser.add_argument('--lr', type=float, required=False,
	                    default=0.01,
	                    help='Starting learning rate')
	parser.add_argument('--datadir', type=str, required=False,
	                    default=osp.join('/home/GAIT_local/SSD', 'TUM_GAID_tf'),
	                    help="Full path to data directory")
	parser.add_argument('--experdir', type=str, required=True,
	                    default=osp.join(homedir, 'experiments', 'tumgaid_multimodal'),
	                    help="Base path to save results of training")
	parser.add_argument('--prefix', type=str, required=True,
	                    default="demo",
	                    help="String to prefix experiment directory name.")
	parser.add_argument('--bs', type=int, required=False,
	                    default=150,
	                    help='Batch size')
	parser.add_argument('--epochs', type=int, required=False,
	                    default=150,
	                    help='Maximum number of epochs')
	parser.add_argument('--extraepochs', type=int, required=False,
	                    default=50,
	                    help='Extra number of epochs to add validation data')
	parser.add_argument('--nclasses', type=int, required=True,
	                    default=150,
	                    help='Maximum number of epochs')
	parser.add_argument('--model_version', type=str, required=False,
	                    default='iwann',
	                    help='Model version. [iwann, bmvc]')
	parser.add_argument('--tdim', type=int, required=False,
	                    default=50,
	                    help='Number of dimensions in 3rd axis time. E.g. OF=50')
	parser.add_argument('--optimizer', type=str, required=False,
	                    default="SGD",
	                    help="Optimizer: SGD, Adam, AMSGrad")
	parser.add_argument('--mod', type=str, required=False,
	                    default="of",
	                    help="Input modality: of, gray, depth")
	parser.add_argument('--initnet', type=str, required=False,
	                    default="",
	                    help="Path to net to initialize")
	parser.add_argument("--verbose", type=int,
	                    nargs='?', const=False, default=1,
	                    help="Whether to enable verbosity of output")

	# Pruning
	parser.add_argument('--pruning', default=False, action='store_true', help="Apply pruning")
	parser.add_argument('--frequency', type=int, required=False,
						default='100',
						help='Only apply pruning every frequency steps')
	parser.add_argument('--begin_step', type=int, required=False,
						default='2000',
						help='Step at which to begin pruning')
	parser.add_argument('--target_sparsity', type=float, required=False,
						default='0.9',
						help='A scalar float representing the target sparsity value')
	parser.add_argument('--pstruc', default=False, action='store_true', help="Structural Pruning")

	args = parser.parse_args()
	verbose = args.verbose
	dropout = args.dropout
	datadir = args.datadir
	prefix = args.prefix
	epochs = args.epochs
	extraepochs = args.extraepochs
	batchsize = args.bs
	nclasses = args.nclasses
	model_version = args.model_version
	lr = args.lr
	tdim = args.tdim
	optimizer = args.optimizer
	experdirbase = args.experdir
	modality = args.mod
	use3D = args.use3d
	IS_DEBUG = args.debug
	freeze_all = args.freezeall
	nofreeze = args.nofreeze
	initnet = args.initnet
	pruning = args.pruning
	frequency = args.frequency
	begin_step = args.begin_step
	target_sparsity = args.target_sparsity
	pstruc = args.pstruc

	# Start the processing
	if nclasses == 50:
		# Train as many models as cameras.
		cameras = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
		for camera in cameras:
			cameras_ = cameras.copy()
			cameras_.remove(camera)
			print("Fine tuning with ", cameras_, " cameras")
			final_model, experdir = trainGaitNet(datadir=datadir, experfix=prefix, lr=lr, dropout=dropout,
			                                     experdirbase=experdirbase, nclasses=nclasses, optimizer=optimizer,
			                                     epochs=epochs, batchsize=batchsize, logdir="", modality=modality,
			                                     initnet=initnet, use3D=use3D, freeze_all=freeze_all, nofreeze=nofreeze,
			                                     extra_epochs=extraepochs, model_version=model_version,
			                                     cameras=cameras_,
			                                     sufix=str(camera).zfill(3), verbose=verbose, pruning=pruning,
                                                     frequency=frequency, begin_step=begin_step,
                                                     target_sparsity=target_sparsity, pstruc=pstruc)
			if final_model is not None:
				final_model.save()
	else:
		final_model, experdir = trainGaitNet(datadir=datadir, experfix=prefix, lr=lr, dropout=dropout,
		                                     experdirbase=experdirbase, nclasses=nclasses, optimizer=optimizer,
		                                     epochs=epochs, batchsize=batchsize, logdir="", modality=modality,
		                                     initnet=initnet, use3D=use3D, freeze_all=freeze_all, nofreeze=nofreeze,
		                                     extra_epochs=extraepochs, model_version=model_version,
		                                     verbose=verbose, pruning=pruning,
                                                     frequency=frequency, begin_step=begin_step,
                                                     target_sparsity=target_sparsity, pstruc=pstruc)
		if not pstruc and final_model is not None:
		#if final_model is not None:
			final_model.save()

	print("* End of training: {}".format(experdir))
