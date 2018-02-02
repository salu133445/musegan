from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import scipy.misc
import numpy as np
import tensorflow as tf
from pprint import pprint
import SharedArray as sa
import pickle

from musegan.core import *
from musegan.components import *
from input_data import *
from config import *


from musegan.libs.utils import *

if __name__ == '__main__':

    """ Create TensorFlow Session """

    t_config = TrainingConfig


    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    gen_dir = 'interpolation/try'

    with tf.Session(config=config) as sess:

        # Onebar
        t_config.exp_name = 'exps/onebar_hybrid'
        model = NowbarHybrid(OneBarHybridConfig)
        input_data = InputDataNowBarHybrid(model)


        musegan = MuseGAN(sess, t_config, model)

        musegan.load(musegan.dir_ckpt)

        z_interpolation = dict()

    #################### SINGLE TRACK ###################
        # gen_dir = 'interpolation/gen/intra_4'
        # st_z_inter = np.random.normal(0, 0.1, [64]).astype(np.float32)

        # st_z_intra_inv = np.random.normal(0, 0.1, [64, 64, 4]).astype(np.float32)
        # st_z_intra_v = np.random.normal(0, 0.1, [64, 1]).astype(np.float32)
        # ed_z_intra_v = np.random.normal(0, 0.1, [64, 1]).astype(np.float32)

        # intra_v_list = np.array([st_z_intra_v] + slerp(st_z_intra_v, ed_z_intra_v, steps=62) + [ed_z_intra_v])
        # z_interpolation['inter'] = np.tile(st_z_inter, (64, 1))
        # z_interpolation['intra'] = np.concatenate([st_z_intra_inv, intra_v_list], axis=2)

        # result, eval_result = musegan.gen_test(input_data, is_eval=True, gen_dir=gen_dir, key=None, is_save=True, z=z_interpolation, type_=1)

        # make_gif(os.path.join(gen_dir, 'sample_binary/*.png'), gen_dir= gen_dir)


    # print(z_interpolation['intra'].shape)
    # np.concatenate([ st_z_intra_v], axis=2)
    # z_list_inter = np.array([st_z_inter] +  + [ed_z_inter])
    # z_list_intra = np.array([st_z_intra] + slerp(st_z_intra, ed_z_intra, steps=62) + [ed_z_intra])

        # ################### GRID interpoaltion ###################

        gen_dir = 'interpolation/gen/bilerp'


        # inter_a0 =  np.random.normal(0, 0.1, [64]).astype(np.float32)
        # intra_b0 =  np.random.normal(0, 0.1, [64, 5]).astype(np.float32)

        # inter_a1 = inter_a0 + 0.0005
        # intra_b1 = intra_b0 + 0.0005


        inter_a0 = np.ones([64]).astype(np.float32) * -0.0005
        intra_b0 =  np.ones([64, 5]).astype(np.float32) * -0.0005

        inter_a1 = np.ones([64]).astype(np.float32) * 0.0005
        intra_b1 = np.ones([64, 5]).astype(np.float32) * 0.0005

        grid_list = bilerp(inter_a0, inter_a1, intra_b0, intra_b1, 8)

        z_interpolation['inter'] = np.array([t[0] for t in grid_list])
        z_interpolation['intra'] = np.array([t[1] for t in grid_list])
        # print(inter.shape, intra.shape)

        result, eval_result = musegan.gen_test(input_data, is_eval=True, gen_dir=gen_dir, key=None, is_save=True, z=z_interpolation, type_=1)
        make_gif(os.path.join(gen_dir, 'sample_binary/*.png'), gen_dir= gen_dir)


        # ################### OLD ###################
        # init


        # z_interpolation['inter'] = np.reshape(st_z_inter, (1,64))
        # z_interpolation['intra'] = np.reshape(st_z_intra, (1,64,5))

        # gen_dir = 'interpolation/gen/init'

        # result, eval_result = musegan.gen_test(input_data, is_eval=True, gen_dir=gen_dir, key=None, is_save=True, z=z_interpolation, type_=1)

        # make_gif(os.path.join(gen_dir, 'sample_binary/*.png'), gen_dir= gen_dir)

        # # inter
        # gen_dir = 'interpolation/gen/inter'
        # print(gen_dir)

        # z_interpolation['inter'] = np.tile(st_z_inter, (64, 1))
        # z_interpolation['intra'] = z_list_intra

        # result, eval_result = musegan.gen_test(input_data, is_eval=True, gen_dir=gen_dir, key=None, is_save=True, z=z_interpolation, type_=1)

        # make_gif(os.path.join(gen_dir, 'sample_binary/*.png'), gen_dir= gen_dir)

        # # intra
        # gen_dir = 'interpolation/gen/intra'
        # print(gen_dir)

        # z_interpolation['inter'] = z_list_inter
        # z_interpolation['intra'] = np.tile(st_z_intra, (64, 1, 1))

        # result, eval_result = musegan.gen_test(input_data, is_eval=True, gen_dir=gen_dir, key=None, is_save=True, z=z_interpolation, type_=1)

        # make_gif(os.path.join(gen_dir, 'sample_binary/*.png'), gen_dir= gen_dir)


        # # both
        # gen_dir = 'interpolation/gen/both'
        # print(gen_dir)

        # z_interpolation['inter'] = z_list_inter
        # z_interpolation['intra'] = z_list_intra

        # result, eval_result = musegan.gen_test(input_data, is_eval=True, gen_dir=gen_dir, key=None, is_save=True, z=z_interpolation, type_=1)

        # make_gif(os.path.join(gen_dir, 'sample_binary/*.png'), gen_dir= gen_dir)


