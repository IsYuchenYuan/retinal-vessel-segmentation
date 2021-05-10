###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import numpy as np
from six.moves import configparser
import torch.backends.cudnn as cudnn
import torch.nn as nn
import sys
from ptflops import get_model_complexity_info

sys.path.insert(0, '../../')
sys.path.insert(1, '../')
from lib.help_functions import *

# function to obtain data for training/testing (validation)
from lib.extract_patches import get_data_training

from torch.optim import SGD

import os

from losses import *

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import random

import torch

from losses import LossMulti

import logging

from tensorboardX import SummaryWriter

from model import Net
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
import torch.nn.functional as F

OUTPUT_DIR = "aug200_4_4000_256x256_adam0.0005_steplr20_checkpoint_log"

def trainlog(logfilepath, head='%(message)s'):
    logger = logging.getLogger('mylogger')
    logging.basicConfig(filename=logfilepath, level=logging.INFO, format=head)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========  Load settings from Config file
config = configparser.RawConfigParser()
config.read('../../configuration_stare.txt')

# patch to the datasets
path_data = config.get('data paths', 'path_local')
print("path_data",path_data)
# Experiment name
name_experiment = config.get('experiment name', 'name')
# training settings
N_epochs = int(config.get('training settings', 'N_epochs'))


# ========== Define parameters here =============================
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
total_epoch = 300
val_portion = 0.1
LR = 0.0005
LR_MOM = 0.9
POLY_POWER = 0.9
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
batch_size = 4

net = Net(3, 2)
criterion = nn.CrossEntropyLoss()

def poly_lr_scheduler(opt, init_lr, iter, max_iter, power):
    new_lr = init_lr * (1 - float(iter) / max_iter) ** power
    return new_lr
optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

# ============ Load the data and divided in patches
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original="../" + path_data + config.get('data paths', 'train_imgs_original'),
    DRIVE_train_groudTruth="../" + path_data + config.get('data paths', 'train_groundTruth'),  # masks
    patch_height=int(config.get('data attributes', 'patch_height')),
    patch_width=int(config.get('data attributes', 'patch_width')),
    N_subimgs=int(config.get('training settings', 'N_subimgs')),
    inside_FOV=config.getboolean('training settings', 'inside_FOV')
    # select the patches only inside the FOV  (default == True)
)

patches_imgs_valid, patches_masks_valid = get_data_training(
    DRIVE_train_imgs_original="../" + path_data + config.get('data paths', 'test_imgs_original'),
    DRIVE_train_groudTruth="../" + path_data + config.get('data paths', 'test_groundTruth'),  # masks
    patch_height=int(config.get('data attributes', 'patch_height')),
    patch_width=int(config.get('data attributes', 'patch_width')),
    N_subimgs=400,
    inside_FOV=config.getboolean('training settings', 'inside_FOV')
    # select the patches only inside the FOV  (default == True)
)

print("patches_imgs_valid shape", patches_imgs_valid.shape)


class TrainDataset(Dataset):
    """Endovis 2018 dataset."""

    def __init__(self, patches_imgs, patches_masks_train):
        self.imgs = patches_imgs
        self.masks = patches_masks_train

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        tmp = self.masks[idx]
#         print("tmppppppppppppp",tmp.shape)
        tmp = np.squeeze(tmp, 0)
        return torch.from_numpy(self.imgs[idx, ...]).float(), torch.from_numpy(tmp).long()



train_set = TrainDataset(patches_imgs_train, patches_masks_train)
train_loader = DataLoader(train_set, batch_size=batch_size,
                          shuffle=True, num_workers=0)

val_set = TrainDataset(patches_imgs_valid, patches_masks_valid)
val_loader = DataLoader(val_set, batch_size=batch_size,
                        shuffle=True, num_workers=0)

best_loss = np.Inf
if device != 'cpu':
    net.to(device)


def train(epoch, lr):
    logging.info('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    logging.info("Learning rate = %4f\n" % lr)
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs0, outputs1, outputs2, outputs3, outputs0_2, outputs1_2, outputs2_2, outputs3_2, final=net(inputs)
        loss0 = criterion(outputs0, targets)
        loss1 = criterion(outputs1, targets) 
        loss2 = criterion(outputs2, targets) 
        loss3 = criterion(outputs3, targets) 
        loss0_2 = criterion(outputs0_2, targets) 
        loss1_2 = criterion(outputs1_2, targets)
        loss2_2 = criterion(outputs2_2, targets) 
        loss3_2 = criterion(outputs3_2, targets)
        finalloss = criterion(final, targets)
        loss = loss0 + loss1 + loss2 + loss3 + loss0_2 + loss1_2 + loss2_2 + loss3_2 + finalloss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_loss = train_loss / np.float32(len(train_loader))
    logging.info("Epoch %d: Train loss %4f\n" % (epoch, avg_loss))
    return avg_loss

def test(epoch):
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs0, outputs1, outputs2, outputs3, outputs0_2, outputs1_2, outputs2_2, outputs3_2, final=net(inputs)
            loss0 = criterion(outputs0, targets)
            loss1 = criterion(outputs1, targets) 
            loss2 = criterion(outputs2, targets) 
            loss3 = criterion(outputs3, targets) 
            loss0_2 = criterion(outputs0_2, targets) 
            loss1_2 = criterion(outputs1_2, targets)
            loss2_2 = criterion(outputs2_2, targets) 
            loss3_2 = criterion(outputs3_2, targets)
            finalloss = criterion(final, targets)
            loss = loss0 + loss1 + loss2 + loss3 + loss0_2 + loss1_2 + loss2_2 + loss3_2 + finalloss
            test_loss += loss.item()
        avg_loss = test_loss / np.float32(len(val_loader))
        logging.info("Epoch %d: Valid loss %4f\n" % (epoch, avg_loss))

    return avg_loss

def main():
    global best_loss
    checkpoint_dir = os.path.join("stare",
                                  OUTPUT_DIR,
                                  "checkpoints"
                                  )
    # Path to save log
    log_dir = os.path.join("stare",
                           OUTPUT_DIR,
                           "logs"
                           )
    makedirs(checkpoint_dir)
    print("Checkpoint dst:", checkpoint_dir)
    makedirs(log_dir)
    print("logit dst:", log_dir)
    logfile = '%s/trainlog.log' % log_dir
    trainlog(logfile)
    for epoch in range(start_epoch, total_epoch):

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        train(epoch, lr)
        test_loss = test(epoch)
        # Save checkpoint.
        if test_loss < best_loss:
            logging.info('Saving epoch %d.pth' % (epoch))
            state = {'epoch': epoch,  # 保存的当前轮数
                     'state_dict': net.state_dict(),  # 训练好的参数
                     'optimizer': optimizer.state_dict(),  # 优化器参数,为了后续的resume
                        }
            save_path = os.path.join(checkpoint_dir, 'epoch %d.pth' % (epoch))
            torch.save(state, save_path)
            best_loss = test_loss


if __name__ == '__main__':
    main()
