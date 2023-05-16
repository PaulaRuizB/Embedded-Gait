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

import deepdish as dd
from sklearn.metrics import confusion_matrix
from scipy import stats
from misc.knn import KNN
from data.dataGeneratorGaitSet_new import DataGeneratorGait
from nets.gaitset import MatMul

# --------------------------------
import tensorflow as tf

theSEED = 232323
tf.random.set_seed(theSEED)
config = tf.compat.v1.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # gpu_rate
tf.config.run_functions_eagerly(True)
tf.executing_eagerly()
graph = tf.Graph()
graph.as_default()

session = tf.compat.v1.Session(graph=graph, config=config)
session.as_default()

# --------------------------------
def encodeData(data_generator, model, model_version):
    all_vids = []
    all_gt_labs = []
    all_feats = []
    all_cams = []
    nbatches = len(data_generator.batches_test)
    for bix in range(nbatches):
        print("Encoding ", bix, "/", nbatches, end='\r')
        data, labels, videoId, cams = data_generator.__getitemvideoid__(bix)
        if len(data) > 1:
            data2 = data[0]
        else:
            data2 = data
        for i in range(len(data2)):
            if model_version == "gaitset_transformer" \
                    or model_version == "gaitset_transformer_v2" \
                    or model_version == "gaitset_transformer_reduceChannel" \
                    or model_version == "gaitset_transformer_v2_reduceChannel" \
                    or model_version == "gaitset_transformer_v3" \
                    or model_version == "gaitset_transformer_v4" \
                    or model_version == "gaitset_transformer_v6" \
                    or model_version == "gaitset_transformer_v3_reduceChannel" \
                    or model_version == "gaitset_structural_pruning":

                x1 = np.expand_dims(data2[i], axis=0)
                x2 = np.asarray(data[1][i])
                x2 = np.expand_dims(x2, axis=0)
                feats = model.encode([x1, x2], batch_size=5000)
            elif model_version == "gaitset":
                feats = model.encode(data2, batch_size=5000)
            elif model_version == "gaitset_4dims":
                data2 = np.expand_dims(data2[0][int(len(data2[i]) / 2) - 15:int(len(data2[i]) / 2) - 15 + 30], axis=0)
                feats = model.encode(data2, batch_size=5000)

                ##Measure time and energy

                #total_energy = 0.0
                #time = 0.0
                #for i in range(20):
                #    descriptor_measurer_GPU = EnergyMeter("xavier", 10, 'GPU', '/tmp/', 'descriptor.txt', 0)
                #    descriptor_measurer_GPU.start()
                #    descriptor_measurer_GPU.start_measuring()

                #    ##1 sample

                #    descriptor_measurer_GPU.stop_measuring()

                #    total_energy = total_energy + descriptor_measurer_GPU.total_energy
                #    time = time + descriptor_measurer_GPU.time

                #descriptor_measurer_GPU.finish()

                #print("##########################", "\n", "Descriptor GPU -> Energy: ", (total_energy) / 20.0,
                #      ", Time: ", (time) / 20.0, "##########################")

            else:
                feats = model.encode(np.expand_dims(np.transpose(data[i], [0, 2, 3, 1]), 0),
                                     batch_size=5000)
            all_feats.extend(feats)
            all_vids.append(videoId[i])
            all_gt_labs.append(labels[i])
            all_cams.append(cams[i])

    return all_feats, all_gt_labs, all_vids, all_cams


def test(gallery_feats, gallery_gt_labs, test_feats, test_gt_labs, test_vids, clf, metric, k, p=None):
    pred_labs = clf.predict(gallery_feats, gallery_gt_labs, test_feats, metric, k, p)

    # Summarize per video
    uvids = np.unique(test_vids)

    # Majority voting per video
    all_gt_labs_per_vid = []
    all_pred_labs_per_vid = []
    for vix in uvids:
        idx = np.where(test_vids == vix)[0]

        gt_lab_vid = stats.mode(list(np.asarray(test_gt_labs)[idx]))[0][0]
        pred_lab_vid = stats.mode(list(np.asarray(pred_labs)[idx]))[0][0]

        all_gt_labs_per_vid.append(gt_lab_vid)
        all_pred_labs_per_vid.append(pred_lab_vid)

    all_gt_labs_per_vid = np.asarray(all_gt_labs_per_vid)
    all_pred_labs_per_vid = np.asarray(all_pred_labs_per_vid)

    # At subsequence level
    M = confusion_matrix(test_gt_labs, pred_labs)
    acc = M.diagonal().sum() / len(test_gt_labs)

    # At video level
    Mvid = confusion_matrix(all_gt_labs_per_vid, all_pred_labs_per_vid)
    acc_vid = Mvid.diagonal().sum() / len(all_gt_labs_per_vid)

    return acc, acc_vid


def encode_test(model, datadir, nclasses=50, modality='silhouette', batchsize=128, use3D=False, camera=0, lstm=False,
                nframes=None, sufix="", cut=False, model_version="gaitset", reshape_output=False, embedding=False,
                patch_size=4):
    all_feats_global = []
    all_gt_labs_global = []
    all_vids_global = []
    all_cams_global = []
    if nclasses == 50:
        data_folder_nm = osp.join(datadir,
                                  'tfimdb_casia_b_N050_test_nm05-06_{:03d}_{}25_60x60'.format(camera, modality))
        info_file_nm = osp.join(datadir,
                                'tfimdb_casia_b_N050_test_nm05-06_{:03d}_{}25_60x60.h5'.format(camera, modality))
        dataset_info_nm = dd.io.load(info_file_nm)
        # Find label mapping for training
        if nclasses > 0:
            ulabels = np.unique(dataset_info_nm['label'])
            # Create mapping for labels
            labmap_nm = {}
            for ix, lab in enumerate(ulabels):
                labmap_nm[int(lab)] = ix
        else:
            labmap_nm = None

        testdir = os.path.join(experdir, "results")
        outpath = os.path.join(testdir,
                               "signatures_nm_{:03}_{:03}_{:02}_{}_{}.h5".format(nclasses, camera, nframes, sufix, filename))
        if not os.path.exists(outpath):
            test_generator_nm = DataGeneratorGait(dataset_info_nm, batch_size=batchsize, mode='test', labmap=labmap_nm,
                                                  modality=modality, datadir=data_folder_nm, max_frames=nframes,
                                                  cut=cut, reshape_output=reshape_output, embedding=embedding,
                                                  patch_size=patch_size)
            all_feats, all_gt_labs, all_vids, all_cams = encodeData(test_generator_nm, model, model_version)

            # Save CM
            exper_nm = {}
            exper_nm["feats"] = np.concatenate(np.expand_dims(all_feats, 1), axis=0)
            exper_nm["gtlabs"] = np.asarray(all_gt_labs)
            exper_nm["vids"] = np.asarray(all_vids)
            exper_nm["cams"] = np.asarray(all_cams)

            if outpath is not None:
                dd.io.save(outpath, exper_nm)
                print("Data saved to: " + outpath)
        else:
            exper_nm = dd.io.load(outpath)
            all_feats = exper_nm["feats"]
            all_gt_labs = exper_nm["gtlabs"]
            all_vids = exper_nm["vids"]
            all_cams = exper_nm["cams"]

        all_feats_global.append(all_feats)
        all_gt_labs_global.append(all_gt_labs)
        all_vids_global.append(all_vids)
        all_cams_global.append(all_cams)

        data_folder_bg = osp.join(datadir,
                                  'tfimdb_casia_b_N050_test_bg01-02_{:03d}_{}25_60x60'.format(camera, modality))
        info_file_bg = osp.join(datadir,
                                'tfimdb_casia_b_N050_test_bg01-02_{:03d}_{}25_60x60.h5'.format(camera, modality))
        dataset_info_bg = dd.io.load(info_file_bg)
        # Find label mapping for training
        if nclasses > 0:
            ulabels = np.unique(dataset_info_bg['label'])
            # Create mapping for labels
            labmap_bg = {}
            for ix, lab in enumerate(ulabels):
                labmap_bg[int(lab)] = ix
        else:
            labmap_bg = None

        outpath = os.path.join(testdir,
                               "signatures_bg_{:03}_{:03}_{:02}_{}.h5".format(nclasses, camera, nframes, sufix))
        if not os.path.exists(outpath):
            test_generator_bg = DataGeneratorGait(dataset_info_bg, batch_size=batchsize, mode='test', labmap=labmap_bg,
                                                  modality=modality, datadir=data_folder_bg, max_frames=nframes,
                                                  cut=cut, reshape_output=reshape_output, embedding=embedding,
                                                  patch_size=patch_size)
            all_feats, all_gt_labs, all_vids, all_cams = encodeData(test_generator_bg, model, model_version)

            # Save CM
            exper_bg = {}
            exper_bg["feats"] = np.concatenate(np.expand_dims(all_feats, 1), axis=0)
            exper_bg["gtlabs"] = np.asarray(all_gt_labs)
            exper_bg["vids"] = np.asarray(all_vids)
            exper_bg["cams"] = np.asarray(all_cams)

            if outpath is not None:
                dd.io.save(outpath, exper_bg)
                print("Data saved to: " + outpath)
        else:
            exper_bg = dd.io.load(outpath)
            all_feats = exper_bg["feats"]
            all_gt_labs = exper_bg["gtlabs"]
            all_vids = exper_bg["vids"]
            all_cams = exper_bg["cams"]

        all_feats_global.append(all_feats)
        all_gt_labs_global.append(all_gt_labs)
        all_vids_global.append(all_vids)
        all_cams_global.append(all_cams)

        data_folder_cl = osp.join(datadir,
                                  'tfimdb_casia_b_N050_test_cl01-02_{:03d}_{}25_60x60'.format(camera, modality))
        info_file_cl = osp.join(datadir,
                                'tfimdb_casia_b_N050_test_cl01-02_{:03d}_{}25_60x60.h5'.format(camera, modality))
        dataset_info_cl = dd.io.load(info_file_cl)
        # Find label mapping for training
        if nclasses > 0:
            ulabels = np.unique(dataset_info_cl['label'])
            # Create mapping for labels
            labmap_cl = {}
            for ix, lab in enumerate(ulabels):
                labmap_cl[int(lab)] = ix
        else:
            labmap_cl = None

        outpath = os.path.join(testdir,
                               "signatures_cl_{:03}_{:03}_{:02}_{}.h5".format(nclasses, camera, nframes, sufix))
        if not os.path.exists(outpath):
            test_generator_s = DataGeneratorGait(dataset_info_cl, batch_size=batchsize, mode='test', labmap=labmap_cl,
                                                 modality=modality, datadir=data_folder_cl, max_frames=nframes, cut=cut,
                                                 reshape_output=reshape_output, embedding=embedding,
                                                 patch_size=patch_size)
            all_feats, all_gt_labs, all_vids, all_cams = encodeData(test_generator_s, model, model_version)

            # Save CM
            exper_cl = {}
            exper_cl["feats"] = np.concatenate(np.expand_dims(all_feats, 1), axis=0)
            exper_cl["gtlabs"] = np.asarray(all_gt_labs)
            exper_cl["vids"] = np.asarray(all_vids)
            exper_cl["cams"] = np.asarray(all_cams)

            if outpath is not None:
                dd.io.save(outpath, exper_cl)
                print("Data saved to: " + outpath)
        else:
            exper_cl = dd.io.load(outpath)
            all_feats = exper_cl["feats"]
            all_gt_labs = exper_cl["gtlabs"]
            all_vids = exper_cl["vids"]
            all_cams = exper_cl["cams"]

        all_feats_global.append(all_feats)
        all_gt_labs_global.append(all_gt_labs)
        all_vids_global.append(all_vids)
        all_cams_global.append(all_cams)

    elif nclasses == 5154:
        data_folder = osp.join(datadir, 'tfimdb_ou_mvlp_N05154_test_00_{:03d}_{}25_60x60'.format(camera, modality))
        info_file = osp.join(datadir, 'tfimdb_ou_mvlp_N05154_test_00_{:03d}_{}25_60x60.h5'.format(camera, modality))
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

        testdir = os.path.join(experdir, "results")
        outpath = os.path.join(testdir,
                               "signatures_00_{:05}_{:03}_{:02}_{}_{}.h5".format(nclasses, camera, nframes, sufix, filename))

        if not os.path.exists(outpath):
            test_generator = DataGeneratorGait(dataset_info, batch_size=batchsize, mode='test', labmap=labmap,
                                               modality=modality, datadir=data_folder, max_frames=nframes, cut=cut,
                                               reshape_output=reshape_output, embedding=embedding,
                                               patch_size=patch_size)

            all_feats, all_gt_labs, all_vids, all_cams = encodeData(test_generator, model, model_version)

            # Save CM
            exper = {}
            exper["feats"] = np.concatenate(np.expand_dims(all_feats, 1), axis=0)
            exper["gtlabs"] = np.asarray(all_gt_labs)
            exper["vids"] = np.asarray(all_vids)
            exper["cams"] = np.asarray(all_cams)

            if outpath is not None:
                dd.io.save(outpath, exper)
                print("Data saved to: " + outpath)
        else:
            exper = dd.io.load(outpath)
            all_feats = exper["feats"]
            all_gt_labs = exper["gtlabs"]
            all_vids = exper["vids"]
            all_cams = exper["cams"]

        all_feats_global.append(all_feats)
        all_gt_labs_global.append(all_gt_labs)
        all_vids_global.append(all_vids)
        all_cams_global.append(all_cams)

    else:
        sys.exit(0)

    for i in range(len(all_feats_global)):
        all_feats_global[i] = np.asarray(all_feats_global[i])
        all_gt_labs_global[i] = np.asarray(all_gt_labs_global[i])
        all_vids_global[i] = np.asarray(all_vids_global[i])
        all_cams_global[i] = np.asarray(all_cams_global[i])

    return all_feats_global, all_gt_labs_global, all_vids_global, all_cams_global


def encode_gallery(model, datadir, nclasses=50, modality='silhouette', batchsize=128, use3D=False, lstm=False,
                   nframes=None,
                   cut=False, model_version="gaitset", reshape_output=False, embedding=False, patch_size=4):
    # ---------------------------------------
    # Prepare data
    # ---------------------------------------
    if nclasses == 50:
        cameras = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
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

        testdir = os.path.join(experdir, "results")
        os.makedirs(testdir, exist_ok=True)
        outpath = os.path.join(testdir, "gallery_{:03}_{:02}_knn_{}.h5".format(nclasses, nframes, filename))

        if not os.path.exists(outpath):
            gallery_generator = DataGeneratorGait(dataset_info_gallery, batch_size=batchsize, mode='trainval',
                                                  labmap=labmap_gallery, modality=modality, camera=cameras,
                                                  datadir=data_folder_gallery, augmentation=False, max_frames=nframes,
                                                  cut=cut, reshape_output=reshape_output, embedding=embedding,
                                                  patch_size=patch_size)
            all_feats_gallery, all_gt_labs_gallery, all_vids_gallery, all_cams_gallery = encodeData(gallery_generator,
                                                                                                    model,
                                                                                                    model_version)
            # Save CM
            exper = {}
            exper["feats"] = all_feats_gallery
            exper["gtlabs"] = all_gt_labs_gallery
            exper["vids"] = all_vids_gallery
            exper["cams"] = all_cams_gallery
            print("Saving data:")
            dd.io.save(outpath, exper)
            print("Data saved to: " + outpath)
        else:
            exper = dd.io.load(outpath)
            all_feats_gallery = exper["feats"]
            all_gt_labs_gallery = exper["gtlabs"]
            all_vids_gallery = exper["vids"]
            all_cams_gallery = exper["cams"]

    elif nclasses == 5154:
        cameras = [0, 15, 30, 45, 60, 75, 90, 180, 195, 210, 225, 240, 255, 270]
        data_folder_gallery = osp.join(datadir, 'tfimdb_ou_mvlp_N05154_ft_{}25_60x60'.format(modality))
        info_file_gallery = osp.join(datadir, 'tfimdb_ou_mvlp_N05154_ft_{}25_60x60.h5'.format(modality))
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

        testdir = os.path.join(experdir, "results")
        os.makedirs(testdir, exist_ok=True)
        outpath = os.path.join(testdir, "gallery_{:05}_{:02}_knn_{}.h5".format(nclasses, nframes, filename))

        if not os.path.exists(outpath):
            gallery_generator = DataGeneratorGait(dataset_info_gallery, batch_size=batchsize, mode='trainval',
                                                  labmap=labmap_gallery, modality=modality, camera=cameras,
                                                  datadir=data_folder_gallery, augmentation=False, use3D=use3D,
                                                  lstm=lstm, nframes=nframes, cut=cut, reshape_output=reshape_output,
                                                  embedding=embedding, patch_size=patch_size)
            all_feats_gallery, all_gt_labs_gallery, all_vids_gallery, all_cams_gallery = encodeData(gallery_generator,
                                                                                                    model)

            # Save CM
            exper = {}
            exper["feats"] = all_feats_gallery
            exper["gtlabs"] = all_gt_labs_gallery
            exper["vids"] = all_vids_gallery
            exper["cams"] = all_cams_gallery
            print("Saving data:")
            dd.io.save(outpath, exper)
            print("Data saved to: " + outpath)
        else:
            exper = dd.io.load(outpath)
            all_feats_gallery = exper["feats"]
            all_gt_labs_gallery = exper["gtlabs"]
            all_vids_gallery = exper["vids"]
            all_cams_gallery = exper["cams"]
    else:
        sys.exit(0)

    all_feats_gallery = np.asarray(all_feats_gallery)
    all_gt_labs_gallery = np.asarray(all_gt_labs_gallery)
    all_vids_gallery = np.asarray(all_vids_gallery)
    all_cams_gallery = np.asarray(all_cams_gallery)

    return all_feats_gallery, all_gt_labs_gallery, all_vids_gallery, all_cams_gallery


################# MAIN ################
if __name__ == "__main__":
    import argparse

    # Input arguments
    parser = argparse.ArgumentParser(description='Evaluates a CNN for gait')

    parser.add_argument('--use3d', default=False, action='store_true', help="Use 3D convs in 2nd branch?")
    parser.add_argument('--allcameras', default=False, action='store_true',
                        help="Test with all cameras")

    parser.add_argument('--lstm', default=False, action='store_true',
                        help="Test with all cameras (only for casia)")

    parser.add_argument('--datadir', type=str, required=False,
                        default=osp.join('/home/GAIT_local/SSD', 'TUM_GAID_tf'),
                        help="Full path to data directory")

    parser.add_argument('--model', type=str, required=True,
                        default=osp.join(homedir,
                                         'experiments/tumgaid_mj_tf/tum150gray_datagen_opAdam_bs128_lr0.001000_dr0.30/model-state-0002.hdf5'),
                        help="Full path to model file (DD: .hdf5)")

    parser.add_argument('--bs', type=int, required=False,
                        default=128,
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

    parser.add_argument('--metrics', type=str, required=False,
                        default="L2",
                        help="gray|depth|of|rgb")

    parser.add_argument('--nframes', type=int, required=False,
                        default=25,
                        help='Number Frames')

    parser.add_argument('--model_version', type=str, required=False,
                        default='gaitset',
                        help='Model version. [gaitset, gaitset_transformer]')

    parser.add_argument('--cut', required=False, default=False, action='store_true', help='Remove background?')

    parser.add_argument('--reshape_output', required=False, default=False, action='store_true',
                        help='Reshape output?')

    parser.add_argument('--verbose', default=False, action='store_true', help="Verbose")
    parser.add_argument('--patch_size', type=int, required=False,
                        default=4,
                        help='Patch size')
    parser.add_argument('--attention_mode', type=str, required=False, default='dropout',
                        help='Attention mode: dropout, multihead')
    parser.add_argument('--cross_weight', type=float, required=False, default=0, help='Crossentropy loss weight')

    args = parser.parse_args()
    datadir = args.datadir
    batchsize = args.bs
    nclasses = args.nclasses
    modelpath = args.model
    modality = args.mod
    use3D = args.use3d
    camera = args.camera
    allcameras = args.allcameras
    lstm = args.lstm
    knn = args.knn
    metrics = args.metrics
    metrics = [metrics]
    nframes = args.nframes
    model_version = args.model_version
    cut = args.cut
    verbose = args.verbose
    reshape_output = args.reshape_output
    ps = args.patch_size
    attention_mode = args.attention_mode
    cross_weight = args.cross_weight

    # Call the evaluator
    if allcameras:
        if nclasses < 999:
            test_cameras = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
            gallery_cameras = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
            conds = 3
        else:
            test_cameras = [0, 15, 30, 45, 60, 75, 90, 180, 195, 210, 225, 240, 255, 270]
            gallery_cameras = [0, 15, 30, 45, 60, 75, 90, 180, 195, 210, 225, 240, 255, 270]
            conds = 1
    else:
        test_cameras = [camera]
        if nclasses < 999:
            gallery_cameras = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
            conds = 3
        else:
            gallery_cameras = [0, 15, 30, 45, 60, 75, 90, 180, 195, 210, 225, 240, 255, 270]
            conds = 1

    # ---------------------------------------
    # Load model
    # ---------------------------------------
    experdir, filename = os.path.split(modelpath)

    embedding = False

    if model_version == 'gaitset':
        from nets.gaitset import GaitSet as OurModel

        encode_layer = "encode"
        model = OurModel(experdir)
        input_shape = (None, 64, 44, 1)
        patch_size = 0
    elif model_version == 'gaitset_4dims':
        from nets.gaitset_4dims import GaitSet as OurModel

        encode_layer = "encode"
        model = OurModel(experdir)
        input_shape = (None, 64, 44, 1)
        patch_size = 0
    elif model_version == 'gaitset_pruning':
        from nets.gaitset_pruning import GaitSet as OurModel

        encode_layer = "encode"
        model = OurModel(experdir, batch_size=batchsize)
        embedding = True
        input_shape = (None, 64, 44, 1)
        patch_size = 0
    elif model_version == 'gaitset_structural_pruning':
        from nets.gaitset_structural_pruning import GaitSet as OurModel

        encode_layer = "encode"
        model = OurModel(experdir, batch_size=batchsize)
        embedding = True
        input_shape = (None, 64, 44, 1)
        patch_size = 0
    elif model_version == "gaitset_transformer":
        from nets.gaitset_transformer import GaitSetTransformer as OurModel

        encode_layer = "flatten"
        model = OurModel(experdir, reduce_channel=False, batch_size=1, lstm=lstm)
        input_shape = (64, 64, 1)
    elif model_version == "gaitset_transformer_v2_reduceChannel":
        from nets.gaitset_transformer_V2 import GaitSetTransformer as OurModel

        encode_layer = "flatten"
        model = OurModel(experdir, reduce_channel=True, batch_size=1, lstm=lstm)
        input_shape = (64, 64, 1)
    elif model_version == "gaitset_transformer_v3":
        from nets.gaitset_transformer_V3 import GaitSetTransformer as OurModel

        encode_layer = "flatten"
        model = OurModel(experdir, reduce_channel=False, batch_size=1, lstm=lstm)
        input_shape = (None, 64, 64, 1)
        patch_size = [[ps, ps]]
        embedding = True
    elif model_version == "gaitset_transformer_v3_reduceChannel":
        from nets.gaitset_transformer_V3 import GaitSetTransformer as OurModel

        encode_layer = "flatten"
        model = OurModel(experdir, reduce_channel=True, batch_size=1, lstm=lstm)
        input_shape = (None, 64, 64, 1)
        patch_size = [[ps, ps]]
        embedding = True
    elif model_version == "gaitset_transformer_v4":
        from nets.gaitset_transformer_V4 import GaitSetTransformer as OurModel

        encode_layer = "flatten"
        model = OurModel(experdir, reduce_channel=False, batch_size=1, lstm=lstm)
        input_shape = (None, 64, 64, 1)
        patch_size = [[8, 8], [4, 4], [2, 2]]
        embedding = True
    elif model_version == "gaitset_transformer_v6":
        from nets.gaitset_transformer_V6 import GaitSetTransformer as OurModel

        encode_layer = "flatten"
        model = OurModel(experdir, reduce_channel=False, batch_size=1, lstm=lstm)
        input_shape = (None, 64, 64, 1)
        patch_size = [[ps, ps]]
        embedding = True

    if cut:
        input_shape = (None, 64, 44, 1)

    print(modelpath)
    # model.load(modelpath, encode_layer='flatten')
    optimfun = tf.keras.optimizers.SGD()
    margin = 0.2
    if model_version == 'gaitset_structural_pruning':
        model.build(input_shape=input_shape, optimizer=optimfun, margin=margin,
                    sparsity=0.75, begin_step=40000, frequency=100)
        model.model = tf.keras.models.load_model(modelpath,
                                                 custom_objects={"StructuralPrune": StructuralPrune, "MatMul": MatMul(),
                                                                 "triplet_loss": triplet_loss()}, compile=False)
        model.model = strip_pruning(model.model)
    else:
        optimfun = tf.keras.optimizers.SGD()
        input_shape = (None, 64, 44, 1)
        patch_size = 0

        model.build(input_shape=input_shape, optimizer=optimfun, margin=margin, patchs_sizes=patch_size,
                    attention_mode=attention_mode, crossentropy_weight=cross_weight)

        model.model = tf.keras.models.load_model(modelpath, custom_objects={"MatMul": MatMul, "tf": tf}, compile=False) #partial model
        #model.model.load_weights(modelpath) # complete model
        model.model.summary()
        model.prepare_encode(encode_layer)

    # model.model_encode.load_weights(modelpath)
    # ---------------------------------------
    # Compute ACCs
    # ---------------------------------------
    gallery_feats, gallery_gt_labs, gallery_vids, gallery_cams = encode_gallery(model, datadir, nclasses, modality,
                                                                                batchsize, use3D, lstm, nframes,
                                                                                cut=cut,
                                                                                model_version=model_version,
                                                                                reshape_output=reshape_output,
                                                                                embedding=embedding, patch_size=ps)
    clf = KNN(gpu=0)
    accs_global = np.zeros((conds, len(test_cameras) + 1))
    accs_global_video = np.zeros((conds, len(test_cameras) + 1))
    for test_cam_ix in range(len(test_cameras)):
        test_cam = test_cameras[test_cam_ix]
        test_feats, test_gt_labs, test_vids, test_cams = encode_test(model, datadir, nclasses, modality,
                                                                     batchsize, use3D, test_cam, lstm, nframes, cut=cut,
                                                                     model_version=model_version,
                                                                     reshape_output=reshape_output, embedding=embedding,
                                                                     patch_size=ps)
        for gait_cond_ix in range(conds):
            acc_ = 0
            acc_video = 0
            for gallery_cam_ix in range(len(gallery_cameras)):
                gallery_cam = gallery_cameras[gallery_cam_ix]
                if test_cam != gallery_cam:
                    pos = np.where(np.asarray(gallery_cams) == gallery_cam)[0]
                    gallery_feats_ = gallery_feats[pos, :]
                    gallery_gt_labs_ = gallery_gt_labs[pos]
                    accs = test(gallery_feats_, gallery_gt_labs_, test_feats[gait_cond_ix], test_gt_labs[gait_cond_ix],
                                test_vids[gait_cond_ix], clf, metrics, knn)
                    acc_ = acc_ + accs[0]
                    acc_video = acc_video + accs[1]

            acc_ = acc_ / (len(gallery_cameras) - 1)
            acc_video = acc_video / (len(gallery_cameras) - 1)
            accs_global[gait_cond_ix, test_cam_ix] = acc_
            accs_global_video[gait_cond_ix, test_cam_ix] = acc_video

    accs_global[:, -1] = np.sum(accs_global[:, 0:-1], axis=1) / len(test_cameras)
    accs_global_video[:, -1] = np.sum(accs_global_video[:, 0:-1], axis=1) / len(test_cameras)
    print("Done!")
    print("Subseq acc:")
    print(accs_global)
    print("Video acc:")
    print(accs_global_video)