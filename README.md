# 

# Multi-level Attention Network for Retinal Vessel Segmentation

A implementation of the multi-level attention network for retinal vessel segmentation. 

## Databases

[DRIVE](https://drive.grand-challenge.org),[STARE](http://cecas.clemson.edu/~ahoover/stare/),[CHASE_DB1](https://blogs.kingston.ac.uk/retinal/chasedb1/)

## Methods

#### Pre-processing

Before training, the images of training datasets are pre-processed with the following transformations: 

- Gray-scale conversion
- Standardization
- Contrast-limited adaptive histogram equalization (CLAHE)
- Gamma adjustment

Detailed implementation is showed in `./lib/pre_processing.py`

#### Data-augmentation

For the three databases, various transformations including flipping and rotation are applied on each image in the training set to generate 10 augmented images. 

#### Patch-extraction

The training of the neural network is performed on sub-images (patches) of the pre-processed full images. Each patch, of dimension 256x256, is obtained by randomly selecting its center inside the full image.  For the DRIVE database, a set of 4000 patches is obtained by randomly extracting 20 patches in each of the 200 augmented DRIVE training images. For the STARE database, a set of 5700 patches is obtained by randomly extracting 30 patches in each of the 190 augmented STARE training images (adopt the ’leave-one-out’ technique ). For the CHASE_DB1 database, a set of 8000 patches is obtained by randomly extracting 40 patches in each of the 200 augmented CHASE_DB1 training images. 

Detailed implementation is showed in `./lib/extract_patches.py`

## Experiments

The code is written in Python, it is possible to replicate the experiment by following the guidelines below.

#### Prerequisites

The neural network is developed with the PyTorch library, we refer to the [PyTorch](https://pytorch.org/) for the installation.

The following dependencies are needed:

- numpy >= 1.13.3
- PIL >=4.3.0
- opencv >=2.4.10
- h5py >=2.8.0
- ConfigParser >=3.5.0b2
- scikit-learn >= 0.20.0

#### Training

It is convenient to create HDF5 datasets of the ground truth, masks and images for both training and testing. In the root folder, just run:

```
python prepare_datasets_DRIVE.py
```

The HDF5 datasets for training and testing will be created in the folder `./DRIVE_datasets_training_testing/`.

Now we can configure the experiment. All the settings can be specified in the file `configuration.txt`.

After all the parameters have been configured, you can train the neural network with:

```
python ./src/train.py
```

#### Testing

The parameters for the testing can be specified again in the `configuration.txt` file.

Run testing by:

```
python test.py
```

The following files will be saved in the folder:

- The ROC curve (png)
- The Precision-recall curve (png)
- Picture of all the testing pre-processed images (png)
- Picture of all the corresponding segmentation ground truth (png)
- Picture of all the corresponding segmentation predictions (png)
- One or more pictures including (top to bottom): original pre-processed image, ground truth, prediction
- Report on the performance

#### Statistical significance analysis

use the Wilcoxon’s rank-sum test [1] to conduct the significance analysis

```
python wilcoxon_test.py
```

## Bibliography

[1] Mann, Henry B., and Donald R. Whitney. "On a test of whether one of two random variables is stochastically larger than the other." *The annals of mathematical statistics* (1947): 50-60.

