from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import scipy.misc
import numpy as np
import tensorflow as tf
from pprint import pprint
import SharedArray as sa

from musegan.core import MuseGAN
from config import *
# from config.default_128_r_off_y_off import *


#assign GPU
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_NUM
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

if __name__ == '__main__':


    """ Create TensorFlow Session """
    with tf.Session(config=config) as sess:

        # create MidiNet
        midinet = MuseGAN(sess)

        if IS_TRAIN:
            midinet.init_all()
            if not IS_RETRAIN:
                midinet.load(DIR_CHECKPOINT, component='all')
            midinet.train()
        # else:

        midinet.load(DIR_CHECKPOINT)
        random_picked_sample = np.random.choice(xrange(len(sa.attach(DATA_X_TRA_NAME))), 20000, replace=False)
        x_sample = sa.attach(DATA_X_TRA_NAME)[random_picked_sample]*2. - 1.


        # feed_dict_sample = {midinet.z: z_sample, midinet.x: data_X}


        # z_sample = np.random.normal(0, 0.1, [20000, Z_DIM]).astype(np.float32)
        # z_sample_intra = np.random.normal(0, 0.1, [20000, Z_DIM, 5]).astype(np.float32)
        # feed_dict_sample = {midinet.z_intra:z_sample_intra, midinet.z: z_sample, midinet.x: x_sample }
        # tmp = run_prediction(midinet.G.prediction_binary, feed_dict = feed_dict_sample, batch_size=100)
        # print ('\n\n\n\n\n\n\n\n\n==========================')
        # print (tmp.shape)
        # np.save('gen.npy', tmp)

        dataset_dir = '/home/wayne/NAS/wayne/v3.0/dataset/data_phr/test'
        track_names = ['Bass', 'Drum', 'Guitar', 'Other', 'Piano']
        data_X = []
        for tn in track_names:
            print(tn)
            tmp_data = np.reshape(np.load(os.path.join(dataset_dir, tn +'.npy')),(-1,8, 96,128, 1))
            tmp_data = tmp_data[:, :, :, 24:108, :]
            data_X.append(tmp_data)

        data_X = np.concatenate(data_X, axis=4)
        data_X = data_X.reshape( [-1,96,84,5])
        data_X = data_X * 2 - 1

        z_sample = np.random.normal(0, 0.1, [4064, Z_DIM]).astype(np.float32)
        z_sample_intra = np.random.normal(0, 0.1, [4064, Z_DIM, 5]).astype(np.float32)

        feed_dict_sample = {midinet.z_intra:z_sample_intra, midinet.z: z_sample, midinet.x: data_X}
        tmp = run_prediction(midinet.G.prediction_binary, data_X, feed_dict = feed_dict_sample, batch_size=4, gen_dir='./gen3')
        print ('\n\n\n\n\n\n\n\n\n==========================')
        print (tmp.shape)
        np.save('gen.npy',tmp)