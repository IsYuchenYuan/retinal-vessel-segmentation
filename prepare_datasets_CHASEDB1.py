#==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE database
#
#============================================================

import os
import h5py
import numpy as np
from PIL import Image



def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)



#------------Path of the images --------------------------------------------------------------
#train
original_imgs_train = "./CHASEDB1/training/images/"
groundTruth_imgs_train = "./CHASEDB1/training/1st_manual/"
borderMasks_imgs_train = "./CHASEDB1/training/mask/"
#test
original_imgs_test = "./CHASEDB1/test/images/"
groundTruth_imgs_test = "./CHASEDB1/test/1st_manual/"
borderMasks_imgs_test = "./CHASEDB1/test/mask/"
#---------------------------------------------------------------------------------------------

Nimgs = 20
Nimgs_test = 8
# Nimgs_test=20
channels = 3
height = 960
width = 999
dataset_path = "./CHASEDB1_datasets_training_testing/"

if not os.path.isdir(dataset_path):
    os.mkdir(dataset_path)

def get_datasets(imgs_dir,groundTruth_dir,borderMasks_dir):
    imgs = np.empty((Nimgs,height,width,channels))
    groundTruth = np.empty((Nimgs,height,width))
    border_masks = np.empty((Nimgs,height,width))
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            print("original image: " +files[i])
            img = Image.open(imgs_dir+files[i])
            imgs[i] = np.asarray(img)
            #corresponding ground truth
            groundTruth_name = files[i].split(".")[0] + "_1stHO.png"
            print("ground truth name: " + groundTruth_name)
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)*255
            #corresponding border masks
            border_masks_name = "mask_" + files[i].split(".")[0].split("_")[-1] + ".png"
            print("border masks name: " + border_masks_name)
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            border_masks[i] = np.asarray(b_mask)

    print("imgs max: " +str(np.max(imgs)))
    print("imgs min: " +str(np.min(imgs)))
    print("groundTruth max: " + str(np.max(groundTruth)))
    print("border_masks max: " + str(np.max(border_masks)))
    assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
    assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
    print("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,height,width))
    groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
    border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
    assert(groundTruth.shape == (Nimgs,1,height,width))
    assert(border_masks.shape == (Nimgs,1,height,width))
    return imgs, groundTruth, border_masks

def get_datasets_test(imgs_dir,groundTruth_dir,borderMasks_dir):
    imgs = np.empty((Nimgs_test,height,width,channels))
    groundTruth = np.empty((Nimgs_test,height,width))
    border_masks = np.empty((Nimgs_test,height,width))
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            print("original image: " +files[i])
            img = Image.open(imgs_dir+files[i])
            imgs[i] = np.asarray(img)
            #corresponding ground truth
            groundTruth_name = files[i].split(".")[0] + "_1stHO.png"
            print("ground truth name: " + groundTruth_name)
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)*255
            #corresponding border masks
            border_masks_name = "mask_" + files[i].split(".")[0].split("_")[-1] + ".png"
            print("border masks name: " + border_masks_name)
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            border_masks[i] = np.asarray(b_mask)

    print("imgs max: " +str(np.max(imgs)))
    print("imgs min: " +str(np.min(imgs)))
    print("groundTruth max: " + str(np.max(groundTruth)))
    print("border_masks max: " + str(np.max(border_masks)))
    assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
    assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
    print("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs_test,channels,height,width))
    groundTruth = np.reshape(groundTruth,(Nimgs_test,1,height,width))
    border_masks = np.reshape(border_masks,(Nimgs_test,1,height,width))
    assert(groundTruth.shape == (Nimgs_test,1,height,width))
    assert(border_masks.shape == (Nimgs_test,1,height,width))
    return imgs, groundTruth, border_masks

#getting the training datasets
imgs_train, groundTruth_train, border_masks_train = get_datasets(original_imgs_train,groundTruth_imgs_train,borderMasks_imgs_train)
print("saving train datasets")
write_hdf5(imgs_train, dataset_path + "CHASEDB1_dataset_imgs.hdf5")
write_hdf5(groundTruth_train, dataset_path + "CHASEDB1_dataset_groundTruth.hdf5")
write_hdf5(border_masks_train,dataset_path + "CHASEDB1_dataset_borderMasks.hdf5")

#getting the testing datasets
imgs_test, groundTruth_test, border_masks_test = get_datasets_test(original_imgs_test,groundTruth_imgs_test,borderMasks_imgs_test)
print("saving test datasets")
write_hdf5(imgs_test,dataset_path + "CHASEDB1_dataset_imgs_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + "CHASEDB1_dataset_groundTruth_test.hdf5")
write_hdf5(border_masks_test,dataset_path + "CHASEDB1_dataset_borderMasks_test.hdf5")

