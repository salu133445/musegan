'''
Model Configuration
'''
from __future__ import print_function
import numpy as np
from shutil import copyfile
import os
import tensorflow as tf
print('[*] config...')

def get_colormap():
    colormap = np.array([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.],
                         [1., .5, 0.],
                         [0., .5, 1.]])
    return tf.constant(colormap, dtype=tf.float32, name='colormap')

# class Dataset:
TRACK_NAMES = ['bass', 'drums', 'guitar', 'piano', 'strings']
TRACK_DIM = 5
class NowBarConfig:
    track_names = TRACK_NAMES
    track_dim = len(track_names)
    output_w = 96
    output_h = 84
    acc_idx = 4
    is_bn = True
    z_inter_dim = 64
    z_intra_dim = 64
    colormap = get_colormap()
    lamda = 10
    beta1 = 0.5
    beta2 = 0.9
    lr = 0.001

# general setting
IS_TRAIN = True
IS_RETRAIN = True
GEN_TEST = False

# gpu setting
GPU_NUM = '0'
DATA_X_TRA_NAME = 'tra_X_bars'
DATA_X_VAL_NAME = 'val_X_bars'
DATA_Y_TRA_NAME = 'tra_y_bars'
DATA_Y_VAL_NAME = 'val_y_bars'
BATCH_SIZE = 64
EXP_NAME = 'try2'

# training settings
EPOCH = 10
TRAIN_SIZE = np.inf
SAMPLE_SIZE = 64
PRINT_BATCH = True
SAVE_IMAGE_SUMMARY = False
Z_DIM = 64

if SAMPLE_SIZE >= 64  and SAMPLE_SIZE %8 == 0:
    SAMPLE_SHAPE = [8, SAMPLE_SIZE//8]
elif SAMPLE_SIZE >= 48  and SAMPLE_SIZE %6 == 0:
    SAMPLE_SHAPE = [6, SAMPLE_SIZE//6]
elif SAMPLE_SIZE >= 24 and SAMPLE_SIZE %4 == 0:
    SAMPLE_SHAPE = [4, SAMPLE_SIZE//4]
elif SAMPLE_SIZE >= 15 and SAMPLE_SIZE %3 == 0:
    SAMPLE_SHAPE = [3, SAMPLE_SIZE//3]
elif SAMPLE_SIZE >= 8 and SAMPLE_SIZE %2 == 0:
    SAMPLE_SHAPE = [2, SAMPLE_SIZE//2]

# directory settings
DIR_CHECKPOINT = 'checkpoint'
DIR_SAMPLE = 'samples'
DIR_LOG = 'logs'

# model settings
OUTPUT_BAR = 8

# Directory settings
DIR_CHECKPOINT = os.path.join(EXP_NAME, DIR_CHECKPOINT)
DIR_SAMPLE = os.path.join(EXP_NAME, DIR_SAMPLE)
DIR_LOG = os.path.join(EXP_NAME, DIR_LOG)

if not os.path.exists(DIR_CHECKPOINT):
    os.makedirs(DIR_CHECKPOINT)
if not os.path.exists(DIR_SAMPLE):
    os.makedirs(DIR_SAMPLE)
if not os.path.exists(DIR_LOG):
    os.makedirs(DIR_LOG)

# save config file
copyfile('./config.py', DIR_LOG+'/config.py')