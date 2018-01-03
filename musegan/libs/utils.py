"""
Some codes from https://github.com/Newmu/model_code
"""
from __future__ import division
import math
import json
import random
import scipy.misc
import numpy as np
from time import gmtime, strftime
import write_midi
import sys
import os

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imsave(images, size, path, boarder=3):
    return scipy.misc.imsave(path, merge(images, size, boarder=boarder))

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

def save_bars(bars, size, file_path, ):
    return imsave(to_image_np(bars), size, file_path)

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

def make_gif(images, fname, duration=2, true_image=False):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images)/duration*t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x+1)/2*255).astype(np.uint8)

        clip = mpy.VideoClip(make_frame, duration=duration)
        clip.write_gif(fname, fps = len(images) / duration)

def save_phrase(images, midi_path):
    images_with_pause = images
    images_with_pause_list = []
    for ch_idx in range(images.shape[3]):
        images_with_pause_list.append(images_with_pause[:,:,:,ch_idx].reshape(images_with_pause.shape[0], images_with_pause.shape[1],images_with_pause.shape[2]))
    write_midi.write_piano_rolls_to_midi(images_with_pause_list, program_nums=[0,49,0], is_drum=[False, False, True], filename=midi_path, tempo=80.0)

def gen_phrase(sess, model, config, num_bars=4, from_scratch=False, gen_dir='gen'):
    num_bars = num_bars-1

    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)

    if from_scratch:
        phrase = np.zeros((config.batch_size, 96, 128, 3))
    else:
        print '[*] Reload Data for scratch'
        data_m = np.reshape(np.load('/home/wayne/NAS/wayne/parsing/data/200RC/pruned_m_RC.npy'),(-1,1,96,128))
        data_r = np.reshape(np.load('/home/wayne/NAS/wayne/parsing/data/200RC/pruned_r_RC.npy'),(-1,1,96,128))
        data_d = np.reshape(np.load('/home/wayne/NAS/wayne/parsing/data/200RC/pruned_d_RC.npy'),(-1,1,96,128))
        data_X = np.concatenate((data_m,data_r,data_d),axis = 1)

        data_X = np.transpose(data_X,(0,2,3,1)) #(14371, 96, 128, 3)
        candi_idx = random.sample(range(data_X.shape[0]), config.batch_size)
        phrase = [data_X[candi_idx, :, :, :]]

    save_images(phrase[0][0:24], [3, 8], './{}/test.png'.format(gen_dir))
    save_midis(phrase[0][0:24], './{}/test.mid'.format(gen_dir))

    for idx in range(num_bars):
        bar = gen_bar(sess, model, config, prev_bar = phrase[-1], bar_idx=idx, gen_dir=gen_dir)
        phrase.append(bar)

    for idx in range(24):
        print('gen song:', idx)
        song_list = []
        for bidx in range(num_bars+1):
            song_list.append([phrase[bidx][idx, :, :, :]])
        song_list = np.concatenate(song_list, axis=1)

        save_phrase(song_list, './{}/song_{:02d}.mid'.format(gen_dir, idx))

def gen_bar(sess, model, config, prev_bar, bar_idx, gen_dir):
    z_sample = np.random.normal(0, 0.1, size=(config.batch_size, model.z_dim))
    samples = sess.run(model.sampler, feed_dict={model.z: z_sample, model.prev_bar: prev_bar})
    save_images(samples[0:24], [3, 8], './{}/test_{:02d}.png'.format(gen_dir, bar_idx))
    save_midis(samples[0:24], './{}/test_{:02d}.mid'.format(gen_dir, bar_idx))
    return samples

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

def run_prediction(prediction_op, data_X, feed_dict, batch_size=64, gen_dir=None):
    """ Run prediction for large data using minibatchs """
    if gen_dir is not None:
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)

    first_item = True
    for key, value in feed_dict.iteritems():
        if first_item:
            length = len(value)
            first_item = False
        else:
            if len(value) != length:
                raise ValueError('Inconsistent item length in feed_dict :' + key)

    feed_dict_batch = {}
    num_batch = (length // batch_size)-1
    prediction_list = []

    for batch in xrange(num_batch):
        sys.stdout.write('{0}/{1}\r'.format(batch, num_batch))
        sys.stdout.flush()

        for key, value in feed_dict.iteritems():
            if batch < num_batch-1:
                feed_dict_batch[key] = value[batch*batch_size:(batch+1)*batch_size]
            else:
                feed_dict_batch[key] = value[batch*batch_size:]

        predictions = prediction_op.eval(feed_dict=feed_dict_batch)
        if(gen_dir is not None):
            batch_data = data_X[batch*batch_size:(batch+1)*batch_size,:,:, :]
            save_bars(batch_data, size=SAMPLE_SHAPE, file_path=os.path.join(gen_dir, '{:03d}'.format(batch) +'real_all.png'))
            batch_data[:,:,:,0:4] = 0
            save_bars(predictions , size=SAMPLE_SHAPE, file_path=os.path.join(gen_dir, '{:03d}'.format(batch) +'fake.png'))
            save_bars(batch_data, size=SAMPLE_SHAPE, file_path=os.path.join(gen_dir, '{:03d}'.format(batch) +'real_p.png'))
            save_midis(predictions, file_path=os.path.join(gen_dir, '{:03d}'.format(batch)  +'fake.mid'))

        prediction_list.append(predictions)
    print('Done!!')
    return np.concatenate(prediction_list, axis=0)
