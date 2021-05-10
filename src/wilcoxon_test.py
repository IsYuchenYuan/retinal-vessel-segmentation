###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################

# Python
import numpy as np
import configparser
from sklearn.utils.multiclass import type_of_target
from skimage import morphology, graph
import matplotlib
from random import randint

matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
import torch
import os
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset

# scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import sys

sys.path.insert(0, '../../')
# help_functions.py
from lib.help_functions import *
# extract_patches.py
from lib.extract_patches import recompone
from lib.extract_patches import recompone_overlap
from lib.extract_patches import paint_border
from lib.extract_patches import kill_border
from lib.extract_patches import pred_only_FOV
from lib.extract_patches import get_data_testing
from lib.extract_patches import get_data_testing_overlap,get_data_testing_overlap_wilcoxon
# pre_processing.py
from lib.pre_processing import my_PreProc

# define pyplot parameters
import matplotlib.pylab as pylab


def load_border():
    # corresponding border masks
    border_masks_name = borderMasks_imgs_test
    print("border masks name: " + borderMasks_imgs_test)
    b_mask = Image.open(borderMasks_imgs_test)
    border_masks = np.asarray(b_mask)[np.newaxis, :]

    print("border_masks max: " + str(np.max(border_masks)))
    assert (np.max(border_masks) == 255)
    assert (np.min(border_masks) == 0)

    border_masks = np.reshape(border_masks, (1, 1, height, width))

    assert (border_masks.shape == (1, 1, height, width))
    return border_masks


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["image"][()]


params = {'legend.fontsize': 15,
          'axes.labelsize': 15,
          'axes.titlesize': 15,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15}
pylab.rcParams.update(params)

# ========= CONFIG FILE TO READ FROM =======
config = configparser.RawConfigParser()
config.read('../../configuration.txt')
# ===========================================
# run the training on invariant or local
path_data = config.get('data paths', 'path_local')
path_data = os.path.join("../", path_data)
# original test images (for FOV selection)
DRIVE_test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]
# the border masks provided by the DRIVE
DRIVE_test_border_masks = path_data + config.get('data paths', 'test_border_masks')
test_border_masks = load_hdf5(DRIVE_test_border_masks)
# test_border_masks=load_border()
# dimension of the patches
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))
# the stride in case output with average
stride_height = int(config.get('testing settings', 'stride_height'))
stride_width = int(config.get('testing settings', 'stride_width'))
assert (stride_height < patch_height and stride_width < patch_width)
# model name
name_experiment = config.get('experiment name', 'name')
path_experiment = "./DRIVE/all_wilcoxon_our" + '/'
if not os.path.exists(path_experiment):
    os.makedirs(path_experiment)
# N full images to be predicted
Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
# Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))
# ====== average mode ===========
average_mode = config.getboolean('testing settings', 'average_mode')

check_path = 'epoch 20.pth'
from model import Net
# from ContrastModel.unet1 import UNet
net = Net(3, 2)
# net = UNet(3,2)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


###########test for 1000 times
for i in range(100):


    # ============ Load the data and divide in patches
    patches_imgs_test = None
    new_height = None
    new_width = None
    masks_test = None
    patches_masks_test = None
    if average_mode == True:
        patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap_wilcoxon(
            DRIVE_test_imgs_original=DRIVE_test_imgs_original,  # original
            DRIVE_test_groudTruth=path_data + config.get('data paths', 'test_groundTruth'),  # masks
            Imgs_to_test=int(config.get('testing settings', 'full_images_to_test')),
            patch_height=patch_height,
            patch_width=patch_width,
            stride_height=stride_height,
            stride_width=stride_width
        )
    else:
        patches_imgs_test, patches_masks_test = get_data_testing(
            DRIVE_test_imgs_original=DRIVE_test_imgs_original,  # original
            DRIVE_test_groudTruth=path_data + config.get('data paths', 'test_groundTruth'),  # masks
            Imgs_to_test=int(config.get('testing settings', 'full_images_to_test')),
            patch_height=patch_height,
            patch_width=patch_width,
        )

    # ================ Run the prediction of the patches ==================================

    resume = True

    if device != 'cpu':
        net.to(device)
    #     net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    #     cudnn.benchmark = True

    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(
            'DRIVE/proposed_aug200_4_4000_256x256_adam0.0005_steplr20_checkpoint_log'), 'Error: no checkpoint directory found!'
        path = 'DRIVE/proposed_aug200_4_4000_256x256_adam0.0005_steplr20_checkpoint_log/checkpoints/' + check_path
        print(path)
        checkpoint = torch.load(path)

        #     print(checkpoint['state_dict'].keys())
        net.load_state_dict(checkpoint['state_dict'])


    class TrainDataset(Dataset):
        """Endovis 2018 dataset."""

        def __init__(self, patches_imgs):
            self.imgs = patches_imgs

        def __len__(self):
            return self.imgs.shape[0]

        def __getitem__(self, idx):
            return torch.from_numpy(self.imgs[idx, ...]).float()


    batch_size = 4

    test_set = TrainDataset(patches_imgs_test)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    preds = []

    # net.eval()
    for batch_idx, inputs in enumerate((test_loader)):
        #     start = time.time()
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = torch.nn.functional.softmax(outputs[8], dim=1)
#         outputs = torch.nn.functional.softmax(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 1)
        shape = list(outputs.shape)
        outputs = outputs.view(-1, shape[1] * shape[2], 2)
        outputs = outputs.data.cpu().numpy()

        preds.append(outputs)

    predictions = np.concatenate(preds, axis=0)
    print("Predictions finished")
    # ===== Convert the prediction arrays in corresponding images
    pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "original")
    # print("pred_patches",pred_patches[0][0][120])
    # ========== Elaborate and visualize the predicted images ====================
    pred_imgs = None
    orig_imgs = None
    gtruth_masks = None
    if average_mode == True:
        pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)  # predictions
        orig_imgs = my_PreProc(test_imgs_orig[0:pred_imgs.shape[0], :, :, :])  # originals
        orig_imgs = np.mean(orig_imgs, axis=1)
        orig_imgs = np.reshape(orig_imgs, (orig_imgs.shape[0], 1, orig_imgs.shape[1], orig_imgs.shape[2]))
        gtruth_masks = masks_test  # ground truth masks
    else:
        pred_imgs = recompone(pred_patches, 13, 12)  # predictions
        orig_imgs = recompone(patches_imgs_test, 13, 12)  # originals
        gtruth_masks = recompone(patches_masks_test, 13, 12)  # masks
    # apply the DRIVE masks on the repdictions #set everything outside the FOV to zero!!
    kill_border(pred_imgs, test_border_masks)  # DRIVE MASK  #only for visualization
    ## back to original dimensions
    orig_imgs = orig_imgs[:, :, 0:full_img_height, 0:full_img_width]
    pred_imgs = pred_imgs[:, :, 0:full_img_height, 0:full_img_width]
    gtruth_masks = gtruth_masks[:, :, 0:full_img_height, 0:full_img_width]
    print("Orig imgs shape: " + str(orig_imgs.shape))
    print("pred imgs shape: " + str(pred_imgs.shape))
    # print("pred imgs : " ,pred_imgs[0][0][250])
    print("Gtruth imgs shape: " + str(gtruth_masks.shape))

    # ====== Evaluate the results
    print("\n\n========  Evaluate the results =======================")
    # predictions only inside the FOV
    y_scores, y_true = pred_only_FOV(pred_imgs, gtruth_masks, test_border_masks)  # returns data only inside the FOV
    print("Calculating results only inside the FOV:")
    print("y scores pixels: " + str(
        y_scores.shape[0]) + " (radius 270: 270*270*3.14==228906), including background around retina: " + str(
        pred_imgs.shape[0] * pred_imgs.shape[2] * pred_imgs.shape[3]) + " (584*565==329960)")
    print("y true pixels: " + str(
        y_true.shape[0]) + " (radius 270: 270*270*3.14==228906), including background around retina: " + str(
        gtruth_masks.shape[2] * gtruth_masks.shape[3] * gtruth_masks.shape[0]) + " (584*565==329960)")


    # # Area under the ROC curve
    fpr, tpr, thresholds = roc_curve((y_true), y_scores)
    AUC_ROC = roc_auc_score(y_true, y_scores)
    # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
    print("\nArea under the ROC curve: " + str(AUC_ROC))

    
    # Confusion matrix
    threshold_confusion = 0.5
    print("\nConfusion matrix:  Costum threshold (for positive) of " + str(threshold_confusion))
    y_pred = np.empty((y_scores.shape[0]))
    for i in range(y_scores.shape[0]):
        if y_scores[i] >= threshold_confusion:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)
    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    print("Global Accuracy: " + str(accuracy))
    specificity = 0
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    print("Specificity: " + str(specificity))
    sensitivity = 0
    if float(confusion[1, 1] + confusion[1, 0]) != 0:
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    print("Sensitivity: " + str(sensitivity))
    precision = 0
    if float(confusion[1, 1] + confusion[0, 1]) != 0:
        precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
    print("Precision: " + str(precision))

    # Jaccard similarity index
    jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
    print("\nJaccard similarity score: " + str(jaccard_index))

    # F1 score
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    print("\nF1 score (F-measure): " + str(F1_score))

    # Save the results
    file_perf = open(path_experiment + 'performances2.txt', 'a')
    # file_perf.write("FPS: %f"%(fps))
    file_perf.write("" + str(AUC_ROC)
                    + "\t" + str(F1_score)
                    + "\t"
                    + str(confusion)
                    + "\t" + str(accuracy)
                    + "\t" + str(sensitivity)
                    + "\t" + str(specificity)
                    + "\n"
                    )
    file_perf.close()
    
    
