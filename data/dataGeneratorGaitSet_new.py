import os
import numpy as np
import random
import deepdish as dd
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from operator import itemgetter
import copy
import gc
import cv2
import imutils

random.seed(10)


class DataGeneratorGait(keras.utils.Sequence):
    """
	A class used to generate data for training/testing CNN gait nets

	Attributes
	----------
	dim : tuple
		dimensions of the input data
	n_classes : int
		number of classes
	...
	"""

    def __init__(self, dataset_info, batch_size=128, mode='train', balanced_classes=True, labmap=[],
                 modality='silohuette',
                 camera=None, datadir="/path_dataset/tfimdb_tum_gaid_N150_train_of25_60x60/",
                 augmentation=True,
                 keep_data=False, mirror=False, resize=False, max_frames=30, test_step=5, p=8, k=16, cut=False,
                 diffFrames=False,
                 reshape_output=False, embedding=False, patch_size=4, crossentropy_loss=False):
        'Initialization'
        self.balanced = balanced_classes
        self.camera = camera
        self.datadir = datadir
        self.empty_files = []
        self.batches = []
        self.batches_test = []
        self.modality = modality
        self.mode = mode
        self.keep_data = keep_data
        self.mirror = mirror
        self.resize = resize
        self.data = {}
        self.epoch = 0
        self.test_step = test_step
        self.p = int(p)
        self.k = int(k)
        self.different_frames = diffFrames
        self.reshape_output = reshape_output
        self.embedding = embedding
        self.patch_size = patch_size
        self.crossentropy_loss = crossentropy_loss

        if mode == 'train':
            self.set = 1
            self.augmentation = augmentation
        elif mode == 'val':
            self.set = 2
            self.augmentation = False
        elif mode == 'trainval' or mode == 'trainvaltest':
            self.set = -1
            self.augmentation = augmentation
            self.shuffle_data = False
        else:
            self.set = 3
            self.augmentation = False

        if mode == 'train':
            ulabs = np.unique(dataset_info['label'])
            nval = int(len(ulabs) * 0.05)
            pos = np.where(dataset_info['label'] < ulabs[-nval])[0]
            self.gait = dataset_info['gait'][pos]
            self.file_list = list(itemgetter(*list(pos))(dataset_info['file']))
            self.labels = dataset_info['label'][pos]
            self.videoId = dataset_info['videoId'][pos]
            self.indexes = np.arange(len(pos))
        elif mode == 'val' or mode == 'valtest':
            ulabs = np.unique(dataset_info['label'])
            nval = int(len(ulabs) * 0.05)
            pos = np.where(dataset_info['label'] >= ulabs[-nval])[0]
            self.gait = dataset_info['gait'][pos]
            self.file_list = list(itemgetter(*list(pos))(dataset_info['file']))
            self.labels = dataset_info['label'][pos]
            self.videoId = dataset_info['videoId'][pos]
            self.indexes = np.arange(len(pos))
        elif mode == 'test':
            pos = np.where(dataset_info['set'] == self.set)[0]
            self.gait = dataset_info['gait'][pos]
            self.file_list = list(itemgetter(*list(pos))(dataset_info['file']))
            self.labels = dataset_info['label'][pos]
            self.videoId = dataset_info['videoId'][pos]
            self.indexes = np.arange(len(pos))
        else:
            self.gait = dataset_info['gait']
            self.file_list = dataset_info['file']
            self.labels = dataset_info['label']
            self.videoId = dataset_info['videoId']
            self.indexes = np.arange(len(self.labels))

        if len(np.unique(dataset_info['label'])) == 74:
            # Remove subject 5
            pos = np.where(self.labels != 5)[0]
            self.gait = self.gait[pos]
            self.file_list = list(itemgetter(*list(pos))(self.file_list))
            self.labels = self.labels[pos]
            self.videoId = self.videoId[pos]
            self.indexes = np.arange(len(pos))

        if self.mode == 'valtest':
            # We have to change the mode to run a normal test for the acc callback.
            self.mode = 'test'

        nclasses = len(np.unique(dataset_info['label']))
        self.ulabs = np.unique(self.labels)

        if nclasses == 150 or nclasses == 155 or nclasses == 16 or nclasses == 500:
            self.camera = None
            self.cameras = None
        else:
            if "cam" in dataset_info.keys():
                self.cameras = dataset_info['cam']
            else:
                self.cameras = self.__derive_camera()

        self.max_frames = max_frames
        self.__remove_empty_files()
        if self.camera is not None:
            self.__keep_camera_samples()

        self.batch_size = np.min((batch_size, len(self.file_list)))  # Deal with less number of samples than batch size

        self.compressFactor = dataset_info['compressFactor']
        self.cut = cut
        if self.modality == 'of':
            if self.cut:
                self.dim = (2, 64, 44)
            else:
                self.dim = (2, 64, 64)
            self.withof = True
        elif self.modality == 'rgb':
            self.dim = (3, 64, 64)
            self.withof = False
        else:
            if self.cut:
                self.dim = (1, 64, 44)
            else:
                self.dim = (1, 64, 64)
            self.withof = False
            self.keep_data = True
        self.ugaits = np.unique(self.gait)
        self.labmap = labmap
        self.list_IDs = []
        self.random_frames = max_frames
        self.__prepare_file_list()
        self.on_epoch_end()

        self.img_gen = self.__transgenerator()

    def __len__(self):
        'Number of batches per epoch'
        return int(np.ceil(len(self.indexes) / np.float(self.batch_size)))

    def __getitem__(self, index):
        """Generate one batch of data"""
        if self.different_frames:
            self.random_frames = np.random.randint(low=1, high=self.max_frames, size=1)[0]  # Number of frames
        else:
            self.random_frames = self.max_frames

        # Generate data
        X, y = self.__data_generation(self.batches[index])

        if self.crossentropy_loss:
            y = [y, y]

        original_size = X.shape
        if self.reshape_output:
            X = np.reshape(X, (self.batch_size * self.random_frames, X.shape[2], X.shape[3], X.shape[4]))

        if self.embedding:
            # npatchs = 16 / self.patch_size
            # positions = np.arange(start=0, stop=npatchs*npatchs*self.random_frames, step=1)
            # d = self.patch_size * self.patch_size * 128
            # k = np.arange(start=0, stop=d / 2, step=1)
            # w_k = np.repeat(np.expand_dims(np.asarray(1 / (np.power(10000, 2 * k / d))),
            # 							   axis=0), axis=0, repeats=positions.shape[0])
            # t = np.repeat(np.expand_dims(positions, axis=1), axis=1, repeats=d / 2)
            # #w_k = tf.math.multiply(w_k, t)
            # w_k = w_k*t
            #
            # #sin = tf.math.sin(w_k)
            # #cos = tf.math.cos(w_k)
            # sin = np.sin(w_k)
            # cos = np.cos(w_k)
            #
            # p = np.dstack((sin, cos)).reshape(sin.shape[0], -1)
            # p = np.expand_dims(p,axis=0)
            p = np.repeat(self.random_frames, self.batch_size, axis=0)
            return [X, p], y
        else:
            return X, y

    def __getitemvideoid__(self, index):
        """Generate one batch of data"""

        # Generate data
        X, y = self.__data_generation(self.batches_test[index])
        videoId = np.asarray(self.videoId)[self.batches_test[index]]

        if self.cameras is not None:
            cameras = np.asarray(self.cameras)[self.batches_test[index]]
        else:
            cameras = None

        '''if isinstance(X, list):
			videoIdFinal = []
			camerasFinal = []
			for i in range(len(X)):
				videoIdFinal.append(np.int32(np.zeros(X[i].shape[0]) + videoId[i]))
				camerasFinal.append(np.int32(np.zeros(X[i].shape[0]) + cameras[i]))

			#videoIdFinal = np.concatenate(videoIdFinal)
			#camerasFinal = np.concatenate(camerasFinal)
			#X = np.concatenate(X, axis=0)
			#y = np.concatenate(y, axis=0)
		else:
			videoIdFinal = videoId
			camerasFinal = cameras'''

        if self.reshape_output:
            if self.mode == 'test' or self.mode == 'trainval' or self.mode == 'trainvaltest':
                X = np.asarray(X)
            else:
                X = np.reshape(X, (self.batch_size * self.random_frames, 64, 64, 1))

        if self.embedding:
            p = []
            for i in range(self.batch_size):
                # 	npatchs = 16 / self.patch_size
                # 	positions = np.arange(start=0, stop=npatchs*npatchs*len(X[i]), step=1)
                # 	d = self.patch_size * self.patch_size * 128
                # 	k = np.arange(start=0, stop=d / 2, step=1)
                # 	w_k = np.repeat(np.expand_dims(np.asarray(1 / (np.power(10000, 2 * k / d))),
                # 							   axis=0), axis=0, repeats=positions.shape[0])
                # 	t = np.repeat(np.expand_dims(positions, axis=1), axis=1, repeats=d / 2)
                # 	#w_k = tf.math.multiply(w_k, t)
                # 	w_k = w_k*t
                #
                # 	#sin = tf.math.sin(w_k)
                # 	#cos = tf.math.cos(w_k)
                # 	sin = np.sin(w_k)
                # 	cos = np.cos(w_k)
                #
                # 	p_ = np.dstack((sin, cos)).reshape(sin.shape[0], -1)
                # 	p_ = np.expand_dims(p_, axis=0)
                # 	p_ = np.repeat(p_, len(X[i]), axis=0)

                p.append(len(X[i]))
            p = np.asarray(p)
            return [X, p], y, videoId, cameras
        else:
            return X, y, videoId, cameras

    def __derive_camera(self):
        cameras = []
        for file in self.file_list:
            parts = file.split("-")
            if np.max(self.ulabs) < 999:
                # 001-nm-01-000.h5
                parts2 = parts[3].split(".")
                cameras.append(int(parts2[0]))
            else:
                # 03314-01-015-01.h5
                parts2 = parts[2].split(".")
                cameras.append(int(parts2[0]))

        return cameras

    def __prepare_file_list(self):
        for l in range(len(self.ulabs)):
            self.list_IDs.append([])
            pos = np.where(self.labels == self.ulabs[l])[0]
            self.list_IDs[l] = pos

    def on_epoch_end(self):
        self.epoch = self.epoch + 1

        # Test batches
        self.batches_test = []
        indexes = np.arange(0, len(self.file_list))
        for i in range(self.__len__()):
            self.batches_test.append([])
            init = i * self.batch_size
            end = min(init + self.batch_size, len(indexes))
            self.batches_test[i].extend(indexes[init:end])
            self.batches_test[i] = np.asarray(self.batches_test[i])

        if self.mode != 'test' and self.mode != "trainval":
            'Updates indexes after each epoch'
            used_samples = []
            used_subjects = np.zeros(len(self.ulabs))
            for l in range(len(self.ulabs)):
                used_samples.append([])
                np.random.shuffle(self.list_IDs[l])
                used_samples[l].append(np.zeros(len(self.list_IDs[l])))

            self.batches = []

            if self.cameras is not None:
                cc = np.asarray(self.cameras)

            # P Classes
            classes_per_batch = min(int(self.p), len(self.ulabs))

            for i in range(self.__len__()):
                self.batches.append([])
                cls_pos = np.where(used_subjects == 0)[0]
                cls_pos = np.random.permutation(cls_pos)
                end_c = min(len(cls_pos), classes_per_batch)
                classes = list(self.ulabs[cls_pos[0:end_c]])
                used_subjects[cls_pos[0:end_c]] = 1
                if len(classes) < classes_per_batch:
                    used_subjects = np.zeros(len(self.ulabs))
                    cls_pos = np.where(used_subjects == 0)[0]
                    cls_pos = np.random.permutation(cls_pos)
                    end_c = min(len(cls_pos), classes_per_batch - len(classes))
                    classes.extend(self.ulabs[cls_pos[0:end_c]])
                    used_subjects[cls_pos[0:end_c]] = 1

                for c in classes:
                    c_idx = np.where(self.ulabs == c)[0][0]
                    rand_samples = self.list_IDs[c_idx]
                    rand_samples = np.random.permutation(rand_samples)
                    self.batches[i].extend(rand_samples[:self.k])
                self.batches[i] = np.asarray(self.batches[i])

        tf.keras.backend.clear_session()
        gc.collect()

    def __load_dd(self, filepath: str, clip_max=0, clip_min=0):
        """
		Loads a dd file with gait samples
		:param filepath: full path to h5 file (deep dish compatible)
		:return: numpy array with data
		"""
        if filepath is None or not os.path.exists(filepath):
            return None

        if self.keep_data:
            if filepath in self.data:
                sample = self.data[filepath]
            else:
                sample = dd.io.load(filepath)
                self.data[filepath] = copy.deepcopy(sample)
        else:
            sample = dd.io.load(filepath)

        if len(sample["data"]) == 0:
            return None

        if sample["compressFactor"] > 1:
            x = np.float32(sample["data"])
            # import pdb; pdb.set_trace()
            # if clip_max > 0:
            #	x[np.abs(x) > clip_max] = 1e-8
            # if clip_min > 0:
            #	x[np.abs(x) < clip_min] = 1e-8
            x = x / sample["compressFactor"]

        # x = x * 0.1  # DEVELOP!
        else:
            if self.modality == 'gray':
                x = (np.float32(sample["data"]) / 255.0) - 0.5
            elif self.modality == 'silhouette':
                x = np.float32(sample["data"]) / 255.0

        if len(np.asarray(x.shape)) == 3:
            x = np.expand_dims(x, axis=1)
        x = np.moveaxis(x, 3, 1)

        return x

    def __remove_empty_files(self):
        gait_ = []
        file_list_ = []
        labels_ = []
        videoId_ = []
        indexes_ = []
        if self.cameras is not None:
            cameras_ = []
        else:
            cameras_ = None
        for i in range(len(self.file_list)):
            data_i = dd.io.load(os.path.join(self.datadir, self.file_list[i]))
            if len(data_i["data"]) >= self.max_frames:
                gait_.append(self.gait[i])
                file_list_.append(self.file_list[i])
                labels_.append(self.labels[i])
                videoId_.append(self.videoId[i])
                indexes_.append(self.indexes[i])
                if self.cameras is not None:
                    cameras_.append(self.cameras[i])

        self.gait = gait_
        self.file_list = file_list_
        self.labels = labels_
        self.videoId = videoId_
        self.indexes = indexes_
        self.cameras = cameras_

    def __keep_camera_samples(self):
        gait_ = []
        file_list_ = []
        labels_ = []
        videoId_ = []
        indexes_ = []
        cameras_ = []
        for i in range(len(self.file_list)):
            for j in range(len(self.camera)):
                cam_str = "{:03d}".format(self.camera[j])
                if cam_str in self.file_list[i] and self.file_list[i] not in file_list_:
                    gait_.append(self.gait[i])
                    file_list_.append(self.file_list[i])
                    labels_.append(self.labels[i])
                    videoId_.append(self.videoId[i])
                    indexes_.append(self.indexes[i])
                    cameras_.append(self.cameras[i])

        self.gait = gait_
        self.file_list = file_list_
        self.labels = labels_
        self.videoId = videoId_
        self.indexes = indexes_
        self.cameras = cameras_

    def __gen_batch(self, list_IDs_temp):
        # Initialization
        if self.mode == 'test' or self.mode == 'trainvaltest' or self.mode == 'trainval':
            # Since we don't know the final number of samples, we use a list + append()
            x = []
            y = []
        else:
            dim0 = len(list_IDs_temp)
            x = np.empty((dim0, self.random_frames, *self.dim), dtype=np.float32)
            y = np.empty((dim0, 1), dtype=np.int32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            filepath = os.path.join(self.datadir, self.file_list[ID])
            label = self.labels[ID]
            if self.labmap:
                label_final = self.labmap[int(label)]
            else:
                label_final = label

            # Data augmentation?
            if self.augmentation and np.random.randint(4) > 0:
                trans = self.img_gen.get_random_transform((self.dim[1], self.dim[2]))
                flip = np.random.randint(2) == 1
            else:
                trans = None
                flip = False

            if self.withof and self.augmentation and np.random.randint(2) == 1:
                clip_max = 2300
                clip_min = 50
            else:
                clip_max = 0
                clip_min = 0

            # Store sample
            x_tmp = self.__load_dd(filepath, clip_max=clip_max, clip_min=clip_min)

            if len(x_tmp) < self.random_frames:
                continue

            if x_tmp is None:
                print("WARN: this shouldn't happen!")
                import pdb
                pdb.set_trace()
            else:
                # Keep X random frames.
                if self.mode != 'test' and self.mode != 'trainval' and self.mode != 'trainvaltest':
                    # Shuffle data
                    np.random.shuffle(x_tmp)

                    # During curriculum learning, random frames is deactivated. We take 25 contiguos frames starting at a random position
                    if x_tmp.shape[0] == self.random_frames:
                        init_pos = 0
                    else:
                        init_pos = np.random.randint(low=0, high=x_tmp.shape[0] - self.random_frames)
                    pos = np.arange(start=init_pos, stop=init_pos + self.random_frames, step=1)

                    x_tmp = x_tmp[pos]
                    if self.cut:
                        x_tmp = x_tmp[:, :, :, 10:54]

                    if self.resize:
                        for im_ix in range(x_tmp.shape[0]):
                            if len(x_tmp.shape) == 4:
                                x_tmp[im_ix, :, :] = cv2.resize(x_tmp[im_ix, :, 10:-10, :], (64, 64))
                            else:
                                x_tmp[im_ix, :, :] = cv2.resize(x_tmp[im_ix, :, 10:-10], (64, 64))

                    if trans is not None:
                        x_tmp = self.__transformsequence(x_tmp, self.img_gen, trans)
                        if flip:
                            x_tmp = self.__mirrorsequence(x_tmp, self.withof, True)
                    if self.mirror and self.mode == 'trainval':
                        x_tmp = self.__mirrorsequence(x_tmp, self.withof, True)

                    x[i] = x_tmp
                    y[i] = label_final
                else:
                    if self.cut:
                        x_tmp = x_tmp[:, :, :, 10:54]
                    # x_tmp = np.moveaxis(x_tmp, 2, -1)
                    x.append(np.stack(x_tmp))
                    # y.append(label_final)
                    y.append(np.asarray(label_final))

        if self.mode != 'test' and self.mode != 'trainval' and self.mode != 'trainvaltest':
            x = np.moveaxis(x, 2, -1)
        else:
            x = np.moveaxis(x, 2, -1)
            y = np.asarray(y)

        return x, y

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        x, y = self.__gen_batch(list_IDs_temp)

        return x, y

    def __mirrorsequence(self, sample, isof=True, copy=True):
        """
		Returns a new variable (previously copied), not in-place!
		:rtype: numpy.array
		:param sample:
		:param isof: boolean. If True, sign of x-channel is changed (i.e. direction changes)
		:return: mirror sample
		"""
        # Make a copy
        if copy:
            newsample = np.copy(sample)
        else:
            newsample = sample

        nt = newsample.shape[0]
        for i in range(nt):
            newsample[i,] = np.flip(newsample[i,], axis=2)
            if isof:
                newsample[i, 0] = -newsample[i, 0]  # Only horizontal component

        return newsample

    def __transforsils(self, sample):
        sample_out = np.zeros_like(sample)
        # This is at frame level.
        mask = np.random.randint(low=1, high=100, size=(sample.shape[-2], sample.shape[-1])) / 100.0
        sample_out[0,] = sample[0,] * mask

        return sample_out

    def __cropsils(self, sample):
        sample_out = np.zeros_like(sample)
        mask = np.ones_like(sample[0,])
        ss = np.copy(sample)
        ss[ss > 0] = 255
        ss = np.uint8(ss)
        if np.random.randint(4) > 0:
            cnts = cv2.findContours(ss[0,].copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            x, y, w, h = cv2.boundingRect(cnts[0])
            min_x = x
            max_x = x + w
            min_y = y
            max_y = y + h
            crop_min_x = np.random.randint(low=min_x, high=max_x)
            crop_max_x = np.random.randint(low=crop_min_x, high=max_x)
            crop_min_y = np.random.randint(low=min_y, high=max_y)
            crop_max_y = np.random.randint(low=crop_min_y, high=max_y)
            mask[crop_min_x:crop_max_x, crop_min_y:crop_max_y] = 0

            sample_out[0,] = sample[0,] * mask

        return sample_out

    def __transformsequence(self, sample, img_gen, transformation):
        sample_out = np.zeros_like(sample)
        # min_v, max_v = (sample.min(), sample.max())
        # abs_max = np.abs(sample).max()
        for i in range(sample.shape[0]):
            I = np.copy(sample[i,])
            It = img_gen.apply_transform(I, transformation)
            if self.modality == 'silhouette':
                It = self.__transforsils(It)
                It = self.__cropsils(It)
            sample_out[i,] = It

        # Fix range if needed
        # if np.abs(sample_out).max() > (3 * abs_max) and self.modality != 'silhouette':  # This has to be normalized
        #	sample_out = (sample_out / 255.0) - 0.5

        return sample_out

    def __transgenerator(self, displace=[-5, -3, 0, 3, 5]):

        if self.modality == 'of' or self.modality == 'depth' or self.modality == 'silhouette':
            ch_sh_range = 0
            br_range = None
        else:
            ch_sh_range = 0.025
            br_range = [0.95, 1.05]

        img_gen = ImageDataGenerator(width_shift_range=displace, height_shift_range=displace,
                                     brightness_range=br_range, zoom_range=0.04,
                                     rotation_range=10, channel_shift_range=ch_sh_range, horizontal_flip=False)

        return img_gen


# ============= MAIN ================
if __name__ == "__main__":
    sess = tf.compat.v1.Session()
    with sess.as_default():
        dataset_info = dd.io.load('/path_dataset/tfimdb_tum_gaid_N150_train_of25_60x60.h5')
        dg = DataGeneratorGait(dataset_info, batch_size=20, mode='train', labmap=None, modality='silhouette',
                               datadir='/path_dataset', camera=None)

    for e in range(0, len(dg)):
        X, Y = dg.__getitem__(e)

    print("Done!")

    for ii in range(X.shape[0]):
        for jj in range(X.shape[1]):
            path = '/tmp/ofs/{:02d}-{:02d}.png'.format(ii, jj)
            im = np.sqrt(np.power(X[ii, jj, 0, :, :], 2) + np.power(X[ii, jj, 1, :, :], 2))
            mi = np.min(im)
            ma = np.max(im)
            im = (im - mi) / (ma - mi)
            cv2.imwrite(path, np.uint8(im * 255))
            cv2.imwrite(path, np.uint8(X[ii, jj, 0, :, :] * 255))