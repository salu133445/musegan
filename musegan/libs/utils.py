"""
Some codes from https://github.com/Newmu/model_code
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import json
import random
import scipy.misc
import numpy as np
from time import gmtime, strftime
from musegan.libs import write_midi
import sys
import os
import imageio
import glob


get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imsave(images, size, path, boarder=3, name='sample', type_=0):
    '''
    type: 0 merge, 1 split

    '''
    scipy.misc.imsave(os.path.join(path, name+'.png'), merge(images, size, boarder=boarder))

    if type_ is 1:
        save_dir = os.path.join(path, name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for idx in range(images.shape[0]):
            scipy.misc.imsave(os.path.join(save_dir, name+'_%d.png'%idx), images[idx, :, :, :])

def merge(images, size, boarder=3):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h*size[0] + boarder*(size[0]-1), w*size[1] + boarder*(size[1]-1), 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        add_h = boarder if j < size[0] else 0
        add_w = boarder if i < size[1] else 0
        img[j*(h+add_h):j*(h+add_h)+h, i*(w+add_w):i*(w+add_w)+w, :] = image

    for i in range(1,size[1]):
        img[:,i*(w+3)-3:i*(w+3)] = [1.0, 1.0, 1.0]
    for j in range(1,size[0]):
        img[j*(h+3)-3:j*(h+3),:] = [1.0, 1.0, 1.0]
    return img

def to_image_np(bars):
    colormap = np.array([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.],
                         [1., .5, 0.],
                         [0., .5, 1.]])
    recolored_bars = np.matmul(bars.reshape(-1, 5), colormap).reshape((bars.shape[0], bars.shape[1], bars.shape[2], 3))
    # recolored_bars = np.zeros((bars.shape[0], bars.shape[1], bars.shape[2], 3))
    # for track_idx in range(bars.shape[-1]):
    #     recolored_bars = recolored_bars + bars[..., track_idx][:, :, :, None]*colormap[track_idx][None, None, None, :]
    return np.flip(np.transpose(recolored_bars, (0, 2, 1, 3)), axis=1)

def save_bars(bars, size, file_path, name='sample', type_=0):
    return imsave(to_image_np(bars), size, file_path, name=name, type_=type_)

def save_midis(bars, file_path):
    padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], 24, bars.shape[3])), bars, np.zeros((bars.shape[0], bars.shape[1], 20, bars.shape[3]))), axis=2)
    pause = np.zeros((bars.shape[0], 96, 128, bars.shape[3]))
    images_with_pause = padded_bars
    images_with_pause = images_with_pause.reshape(-1, 96, padded_bars.shape[2], padded_bars.shape[3])
    images_with_pause_list = []
    for ch_idx in range(padded_bars.shape[3]):
        images_with_pause_list.append(images_with_pause[:,:,:,ch_idx].reshape(images_with_pause.shape[0],  \
                                                        images_with_pause.shape[1], images_with_pause.shape[2]))
    write_midi.write_piano_rolls_to_midi(images_with_pause_list, program_nums=[33,0,25,49,0], is_drum=[False, True, False, False, False],  \
                                                            filename=file_path, tempo=80.0)

def transform(image, npx=64, resize_w=64):
    # npx : # of pixels width/height of image
    return np.array(image)/127.5 - 1.

def make_gif(imgs_filter, gen_dir='./', stop__frame_num=10):
    img_list = glob.glob(imgs_filter)
    images = []
    for filename in img_list:
        images.append(imageio.imread(filename))
    print('%d imgs'% len(img_list))

    stop_frame = np.zeros(images[0].shape)
    images = images + [stop_frame] * stop__frame_num

    imageio.mimsave(os.path.join(gen_dir, 'movie.gif'), images, duration=0.3)


def to_binary_np(bar, threshold=0.0):
    bar_binary = (bar > threshold)
    melody_is_max = (bar[..., 0] == bar[..., 0].max(axis=2, keepdims=True))
    bar_binary[..., 0] = np.logical_and(melody_is_max, bar_binary[..., 0])
    return bar_binary

def to_chroma_np(bar, is_normalize=True):
    chroma = bar.reshape(bar.shape[0], bar.shape[1], 12, 7, bar.shape[3]).sum(axis=3)
    if is_normalize:
        chroma_max = chroma.max(axis=(1, 2, 3), keepdims=True)
        chroma_min = chroma.min(axis=(1, 2, 3), keepdims=True)
        return np.true_divide(chroma + chroma_min, chroma_max - chroma_min + 1e-15)
    else:
        return chroma


def bilerp(a0, a1, b0, b1, steps):

    at = 1 / (steps - 1)
    bt = 1 / (steps - 1)

    grid_list = []
    for aidx in range(0, steps):
        for bidx in range(0, steps):
            a = at * aidx
            b = bt * bidx

            ai = (1-a)*a0 + a*a1
            bi = (1-b)*b0 + b*b1

            grid_list.append((ai, bi))
    return grid_list


def lerp(a, b, steps):
    vec = b - a
    step_vec = vec / (steps+1)
    step_list = []
    for idx in range(1, steps+1):
        step_list.append(a + step_vec*idx)
    return step_list

def slerp(a, b, steps):
    aa =  np.squeeze(a/np.linalg.norm(a))
    bb =  np.squeeze(b/np.linalg.norm(b))
    ttt = np.sum(aa*bb)
    omega = np.arccos(ttt)
    so = np.sin(omega)
    step_deg = 1 / (steps+1)
    step_list = []

    for idx in range(1, steps+1):
        t = step_deg*idx
        tmp = np.sin((1.0-t)*omega) / so * a + np.sin(t*omega)/so * b
        step_list.append(tmp)
    return step_list

def get_sample_shape(sample_size):
    if sample_size >= 64  and sample_size %8 == 0:
        return [8, sample_size//8]
    elif sample_size >= 48  and sample_size %6 == 0:
        return [6,sample_size//6]
    elif sample_size >= 24 and sample_size %4 == 0:
        return [4, sample_size/4]
    elif sample_size >= 15 and sample_size %3 == 0:
        return [3, sample_size//3]
    elif sample_size >= 8 and sample_size %2 == 0:
        return [2, sample_size//2]