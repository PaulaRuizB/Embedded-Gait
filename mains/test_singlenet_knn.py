import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

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

import deepdish as dd
from sklearn.metrics import confusion_matrix
import statistics
from data.dataGenerator import DataGeneratorGait
from nets.single_model import SingleGaitModel
from sklearn.neighbors import KNeighborsClassifier

# --------------------------------
import tensorflow as tf

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

from utils.energy_meter import EnergyMeter

# --------------------------------

def encodeData(data_generator, model):
	all_vids = []
	all_gt_labs = []
	all_feats = []

	#nbatches = len(data_generator) #todo
	nbatches=1 #measure time and energy #todo


	for bix in range(nbatches):
		data, labels, videoId, _ = data_generator.__getitemvideoid__(bix)
		#feats = model.encode(data)

		total_energy = 0.0
		time = 0.0
		print("Starting to measure")
		for i in range(20):

			descriptor_measurer_GPU = EnergyMeter("xavier", 10, 'GPU', '/tmp/', 'descriptor.txt', 0) #todo 10 original
			descriptor_measurer_GPU.start()
			descriptor_measurer_GPU.start_measuring()
			feats = model.encode(data)
			# feats = model.encode(np.expand_dims(data, axis=0))
			#	feats = model.encode(np.expand_dims(data[0], axis=0)) #modified data 1 sample #todo
			descriptor_measurer_GPU.stop_measuring()

			total_energy = total_energy + descriptor_measurer_GPU.total_energy
			time = time + descriptor_measurer_GPU.time
			#print(descriptor_measurer_GPU.time, "\n")

		descriptor_measurer_GPU.finish()
		print("##########################", "\n", "Descriptor GPU -> Energy: ", (total_energy)/20.0,
			  ", Time: ", (time)/20.0, "##########################")


		all_feats.extend(feats)
		all_vids.extend(videoId)
		all_gt_labs.extend(labels[:, 0])

	return all_feats, all_gt_labs, all_vids

def testData(data_generator, model, clf, outpath):

	all_feats, all_gt_labs, all_vids = encodeData(data_generator, model)

	# Save CM
	exper = {}
	exper["feats"] = all_feats
	exper["gtlabs"] = all_gt_labs
	exper["vids"] = all_vids
	dd.io.save(outpath, exper)
	print("Data saved to: " + outpath)


	all_pred_labs = clf.predict(all_feats)


	# Summarize per video
	uvids = np.unique(all_vids)

	# Majority voting per video
	all_gt_labs_per_vid = []
	all_pred_labs_per_vid = []
	for vix in uvids:
		idx = np.where(all_vids == vix)[0]

		try:
			gt_lab_vid = statistics.mode(list(np.asarray(all_gt_labs)[idx]))
		except:
			gt_lab_vid = np.asarray(all_gt_labs)[idx][0]

		try:
			pred_lab_vid = statistics.mode(list(np.asarray(all_pred_labs)[idx]))
		except:
			pred_lab_vid = np.asarray(all_pred_labs)[idx][0]

		all_gt_labs_per_vid.append(gt_lab_vid)
		all_pred_labs_per_vid.append(pred_lab_vid)

	all_gt_labs_per_vid = np.asarray(all_gt_labs_per_vid)
	all_pred_labs_per_vid = np.asarray(all_pred_labs_per_vid)

	# At subsequence level
	M = confusion_matrix(all_gt_labs, all_pred_labs)
	acc = M.diagonal().sum() / len(all_gt_labs)
	print("*** Accuracy [subseq]: {:.2f}".format(acc * 100))

	# At video level
	Mvid = confusion_matrix(all_gt_labs_per_vid, all_pred_labs_per_vid)
	acc_vid = Mvid.diagonal().sum() / len(all_gt_labs_per_vid)
	print("*** Accuracy [video]: {:.2f}".format(acc_vid * 100))


def evalGaitNet(datadir="matimdbtum_gaid_N150_of25_60x60_lite", nclasses=155, initnet="",
                modality='of', batchsize=128, knn=7, use3D=False, camera=0):
	# ---------------------------------------
	# Load model
	# ---------------------------------------

	experdir, filename = os.path.split(initnet)
	model = SingleGaitModel(experdir)

	# Original models
	model.load(initnet)

	# Models 3dto2d
	'''momentum = 0.9
	lr = 0.01
	optimfun = optimizers.SGD(lr=lr, momentum=momentum)
	model.build((25, 60, 60, 2), 4, [(7, 7), (5, 5), (3, 3), (2, 2)], [96, 192, 512, 512], [1, 1, 1, 1],
							1024, 0.00005, 0.4, optimizer=optimfun, nclasses=150,
						 use3D=use3D,
							model_version='bmvc', pruning=False, frequency=100,
							begin_step=1000, target_sparsity=0.9, from3dto2d=True, batchsize=1)

	model.model.load_weights(initnet)
	model.model_encode = tf.keras.Model(model.model.input, model.model.layers[-1].input)
	#model.model.save("/path/3dto2d.h5")'''
	# ---------------------------------------
	# Prepare data
	# ---------------------------------------
	if nclasses == 155:
		data_folder_gallery = osp.join(datadir, 'tfimdb_tum_gaid_N155_ft_{}25_60x60'.format(modality))
		info_file_gallery = osp.join(datadir, 'tfimdb_tum_gaid_N155_ft_{}25_60x60.h5'.format(modality))
		dataset_info_gallery = dd.io.load(info_file_gallery)
		# Find label mapping for training
		if nclasses > 0:
			ulabels = np.unique(dataset_info_gallery['label'])
			# Create mapping for labels
			labmap_gallery = {}
			for ix, lab in enumerate(ulabels):
				labmap_gallery[int(lab)] = ix
		else:
			labmap_gallery = None
		gallery_generator = DataGeneratorGait(dataset_info_gallery, batch_size=batchsize, mode='trainval', labmap=labmap_gallery,
		                                    modality=modality, datadir=data_folder_gallery, augmentation=False, use3D=use3D)

		data_folder_n = osp.join(datadir, 'tfimdb_tum_gaid_N155_test_n05-06_{}25_60x60'.format(modality))
		info_file_n = osp.join(datadir, 'tfimdb_tum_gaid_N155_test_n05-06_{}25_60x60.h5'.format(modality))
		dataset_info_n = dd.io.load(info_file_n)
		# Find label mapping for training
		if nclasses > 0:
			ulabels = np.unique(dataset_info_n['label'])
			# Create mapping for labels
			labmap_n = {}
			for ix, lab in enumerate(ulabels):
				labmap_n[int(lab)] = ix
		else:
			labmap_n = None
		test_generator_n = DataGeneratorGait(dataset_info_n, batch_size=batchsize, mode='test', labmap=labmap_n, modality=modality,
		                                     datadir=data_folder_n, use3D=use3D)

		data_folder_b = osp.join(datadir, 'tfimdb_tum_gaid_N155_test_b01-02_{}25_60x60'.format(modality))
		info_file_b = osp.join(datadir, 'tfimdb_tum_gaid_N155_test_b01-02_{}25_60x60.h5'.format(modality))
		dataset_info_b = dd.io.load(info_file_b)
		# Find label mapping for training
		if nclasses > 0:
			ulabels = np.unique(dataset_info_n['label'])
			# Create mapping for labels
			labmap_b = {}
			for ix, lab in enumerate(ulabels):
				labmap_b[int(lab)] = ix
		else:
			labmap_b = None
		test_generator_b = DataGeneratorGait(dataset_info_b, batch_size=batchsize, mode='test', labmap=labmap_b, modality=modality,
		                                     datadir=data_folder_b, use3D=use3D)

		data_folder_s = osp.join(datadir, 'tfimdb_tum_gaid_N155_test_s01-02_{}25_60x60'.format(modality))
		info_file_s = osp.join(datadir, 'tfimdb_tum_gaid_N155_test_s01-02_{}25_60x60.h5'.format(modality))
		dataset_info_s = dd.io.load(info_file_s)
		# Find label mapping for training
		if nclasses > 0:
			ulabels = np.unique(dataset_info_n['label'])
			# Create mapping for labels
			labmap_s = {}
			for ix, lab in enumerate(ulabels):
				labmap_s[int(lab)] = ix
		else:
			labmap_s = None
		test_generator_s = DataGeneratorGait(dataset_info_s, batch_size=batchsize, mode='test', labmap=labmap_s, modality=modality,
		                                     datadir=data_folder_s, use3D=use3D)
	elif nclasses == 16:
		data_folder_gallery = osp.join(datadir, 'tfimdb_tum_gaid_N016_ft_{}25_60x60'.format(modality))
		info_file_gallery = osp.join(datadir, 'tfimdb_tum_gaid_N016_ft_{}25_60x60.h5'.format(modality))
		dataset_info_gallery = dd.io.load(info_file_gallery)
		# Find label mapping for training
		if nclasses > 0:
			ulabels = np.unique(dataset_info_gallery['label'])
			# Create mapping for labels
			labmap_gallery = {}
			for ix, lab in enumerate(ulabels):
				labmap_gallery[int(lab)] = ix
		else:
			labmap_gallery = None
		gallery_generator = DataGeneratorGait(dataset_info_gallery, batch_size=batchsize, mode='trainval',
		                                      labmap=labmap_gallery, modality=modality,
		                                      datadir=data_folder_gallery, augmentation=False, use3D=use3D)

		data_folder_n = osp.join(datadir, 'tfimdb_tum_gaid_N016_test_n11-12_{}25_60x60'.format(modality))
		info_file_n = osp.join(datadir, 'tfimdb_tum_gaid_N016_test_n11-12_{}25_60x60.h5'.format(modality))
		dataset_info_n = dd.io.load(info_file_n)
		# Find label mapping for training
		if nclasses > 0:
			ulabels = np.unique(dataset_info_n['label'])
			# Create mapping for labels
			labmap_n = {}
			for ix, lab in enumerate(ulabels):
				labmap_n[int(lab)] = ix
		else:
			labmap_n = None
		test_generator_n = DataGeneratorGait(dataset_info_n, batch_size=batchsize, mode='test', labmap=labmap_n, modality=modality,
		                                     datadir=data_folder_n, use3D=use3D)

		data_folder_b = osp.join(datadir, 'tfimdb_tum_gaid_N016_test_b03-04_{}25_60x60'.format(modality))
		info_file_b = osp.join(datadir, 'tfimdb_tum_gaid_N016_test_b03-04_{}25_60x60.h5'.format(modality))
		dataset_info_b = dd.io.load(info_file_b)
		# Find label mapping for training
		if nclasses > 0:
			ulabels = np.unique(dataset_info_n['label'])
			# Create mapping for labels
			labmap_b = {}
			for ix, lab in enumerate(ulabels):
				labmap_b[int(lab)] = ix
		else:
			labmap_b = None
		test_generator_b = DataGeneratorGait(dataset_info_b, batch_size=batchsize, mode='test', labmap=labmap_b, modality=modality,
		                                     datadir=data_folder_b, use3D=use3D)

		data_folder_s = osp.join(datadir, 'tfimdb_tum_gaid_N016_test_s03-04_{}25_60x60'.format(modality))
		info_file_s = osp.join(datadir, 'tfimdb_tum_gaid_N016_test_s03-04_{}25_60x60.h5'.format(modality))
		dataset_info_s = dd.io.load(info_file_s)
		# Find label mapping for training
		if nclasses > 0:
			ulabels = np.unique(dataset_info_n['label'])
			# Create mapping for labels
			labmap_s = {}
			for ix, lab in enumerate(ulabels):
				labmap_s[int(lab)] = ix
		else:
			labmap_s = None
		test_generator_s = DataGeneratorGait(dataset_info_s, batch_size=batchsize, mode='test', labmap=labmap_s, modality=modality,
		                                     datadir=data_folder_s, use3D=use3D)
	elif nclasses == 50:
		data_folder_gallery = osp.join(datadir, 'tfimdb_casia_b_N050_ft_{}25_60x60'.format(modality))
		info_file_gallery = osp.join(datadir, 'tfimdb_casia_b_N050_ft_{}25_60x60.h5'.format(modality))
		dataset_info_gallery = dd.io.load(info_file_gallery)

		# Find label mapping for training
		if nclasses > 0:
			ulabels = np.unique(dataset_info_gallery['label'])
			# Create mapping for labels
			labmap_gallery = {}
			for ix, lab in enumerate(ulabels):
				labmap_gallery[int(lab)] = ix
		else:
			labmap_gallery = None
		cameras = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
		cameras_ = cameras.remove(camera)
		gallery_generator = DataGeneratorGait(dataset_info_gallery, batch_size=batchsize, mode='trainval',
		                                      labmap=labmap_gallery, modality=modality, camera=cameras_,
		                                      datadir=data_folder_gallery, augmentation=False, use3D=use3D)

		data_folder_n = osp.join(datadir, 'tfimdb_casia_b_N050_test_nm05-06_{:03d}_{}25_60x60'.format(camera, modality))
		info_file_n = osp.join(datadir, 'tfimdb_casia_b_N050_test_nm05-06_{:03d}_{}25_60x60.h5'.format(camera, modality))
		dataset_info_n = dd.io.load(info_file_n)
		# Find label mapping for training
		if nclasses > 0:
			ulabels = np.unique(dataset_info_n['label'])
			# Create mapping for labels
			labmap_n = {}
			for ix, lab in enumerate(ulabels):
				labmap_n[int(lab)] = ix
		else:
			labmap_n = None
		test_generator_n = DataGeneratorGait(dataset_info_n, batch_size=batchsize, mode='test', labmap=labmap_n, modality=modality,
		                                     datadir=data_folder_n, use3D=use3D)

		data_folder_b = osp.join(datadir, 'tfimdb_casia_b_N050_test_bg01-02_{:03d}_{}25_60x60'.format(camera, modality))
		info_file_b = osp.join(datadir, 'tfimdb_casia_b_N050_test_bg01-02_{:03d}_{}25_60x60.h5'.format(camera, modality))
		dataset_info_b = dd.io.load(info_file_b)
		# Find label mapping for training
		if nclasses > 0:
			ulabels = np.unique(dataset_info_n['label'])
			# Create mapping for labels
			labmap_b = {}
			for ix, lab in enumerate(ulabels):
				labmap_b[int(lab)] = ix
		else:
			labmap_b = None
		test_generator_b = DataGeneratorGait(dataset_info_b, batch_size=batchsize, mode='test', labmap=labmap_b, modality=modality,
		                                     datadir=data_folder_b, use3D=use3D)

		data_folder_s = osp.join(datadir, 'tfimdb_casia_b_N050_test_cl01-02_{:03d}_{}25_60x60'.format(camera, modality))
		info_file_s = osp.join(datadir, 'tfimdb_casia_b_N050_test_cl01-02_{:03d}_{}25_60x60.h5'.format(camera, modality))
		dataset_info_s = dd.io.load(info_file_s)
		# Find label mapping for training
		if nclasses > 0:
			ulabels = np.unique(dataset_info_n['label'])
			# Create mapping for labels
			labmap_s = {}
			for ix, lab in enumerate(ulabels):
				labmap_s[int(lab)] = ix
		else:
			labmap_s = None
		test_generator_s = DataGeneratorGait(dataset_info_s, batch_size=batchsize, mode='test', labmap=labmap_s, modality=modality,
		                                     datadir=data_folder_s, use3D=use3D)
	else:
		sys.exit(0)

	# ---------------------------------------
	# Test data
	# ---------------------------------------
	all_feats_gallery, all_gt_labs_gallery, all_vids_gallery = encodeData(gallery_generator, model)
	clf = KNeighborsClassifier(n_neighbors=knn) #CPU
	clf.fit(all_feats_gallery, all_gt_labs_gallery)

	print("Evaluating KNN - N...")
	testdir = os.path.join(experdir, "results")
	os.makedirs(testdir, exist_ok=True)
	outpath = os.path.join(testdir, "results_knn_{}_nm_{}_{}.h5".format(knn, nclasses, camera))
	testData(test_generator_n, model, clf, outpath)
	print("Evaluating KNN - B...")
	outpath = os.path.join(testdir, "results_knn_{}_bg_{}_{}.h5".format(knn, nclasses, camera))
	testData(test_generator_b, model, clf, outpath)
	print("Evaluating KNN - S...")
	outpath = os.path.join(testdir, "results_knn_{}_cl_{}_{}.h5".format(knn, nclasses, camera))
	testData(test_generator_s, model, clf, outpath)


################# MAIN ################
if __name__ == "__main__":
	import argparse

	# Input arguments
	parser = argparse.ArgumentParser(description='Evaluates a CNN for gait')

	parser.add_argument('--use3d', default=False, action='store_true', help="Use 3D convs in 2nd branch?")

	parser.add_argument('--allcameras', default=False, action='store_true', help="Test with all cameras (only for casia)")

	parser.add_argument('--datadir', type=str, required=False,
	                    default=osp.join('/path_dataset', 'TUM_GAID_tf'),
	                    help="Full path to data directory")

	parser.add_argument('--model', type=str, required=True,
	                    default=osp.join(homedir,
	                                     'experiments/tumgaid_mj_tf/tum150gray_datagen_opAdam_bs128_lr0.001000_dr0.30/model-state-0002.hdf5'),
	                    help="Full path to model file (DD: .hdf5)")

	parser.add_argument('--bs', type=int, required=False,
	                    default=1,
	                    help='Batch size')

	parser.add_argument('--nclasses', type=int, required=True,
	                    default=155,
	                    help='Maximum number of epochs')

	parser.add_argument('--knn', type=int, required=True,
	                    default=7,
	                    help='Number of noighbours')

	parser.add_argument('--camera', type=int, required=False,
	                    default=90,
	                    help='Camera')

	parser.add_argument('--mod', type=str, required=False,
	                    default="of",
	                    help="gray|depth|of|rgb")

	args = parser.parse_args()
	datadir = args.datadir
	batchsize = args.bs
	nclasses = args.nclasses
	modelpath = args.model
	modality = args.mod
	knn = args.knn
	use3D = args.use3d
	camera = args.camera
	allcameras = args.allcameras

	# Call the evaluator
	cameras = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
	if allcameras:
		for cam in cameras:
			evalGaitNet(datadir=datadir, nclasses=nclasses, initnet=modelpath,
			            modality=modality, batchsize=batchsize, knn=knn, use3D=use3D, camera=cam)
	else:
		evalGaitNet(datadir=datadir, nclasses=nclasses, initnet=modelpath,
		            modality=modality, batchsize=batchsize, knn=knn, use3D=use3D, camera=camera)

	print("Done!")

