from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from musegan.libs.utils import *


def get_placeholder(default_tensor=None, shape=None, name=None):
    """Return a placeholder_wirh_default if default_tensor given, otherwise a new placeholder is created and return"""
    if default_tensor is not None:
        return default_tensor
    else:
        if shape is None:
            raise ValueError('One of default_tensor and shape must be given')
        return tf.placeholder(tf.float32, shape=shape, name=name)

def merge_summaries(scope):
    """Merge the summaries within the given scope and return the merged summary"""
    return tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope))

def batch_norm(tensor_in, apply=True):
    """
    Apply a batch normalization layer on the input tensor and return the resulting tensor.

    Args:
        tensor_in (tensor): The input tensor.
        apply (bool): True to apply. False to bypass batch normalization. Defaults to True.

    Returns:
        tensor: The resulting tensor.

    """
    if tensor_in is not None and apply:
        return tf.contrib.layers.batch_norm(tensor_in, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)
    else:
        return tensor_in

def lrelu(tensor_in, alpha=0.2):
    """Apply a leaky ReLU layer on the input tensor and return the resulting tensor. (alpha defaults to 0.2)"""
    if tensor_in is not None:
        return tf.maximum(tensor_in, alpha*tensor_in)
    else:
        return tensor_in

def relu(tensor_in):
    """Apply a ReLU layer on the input tensor and return the resulting tensor."""
    if tensor_in is not None:
        return tf.nn.relu(tensor_in)
    else:
        return tensor_in

def concat_cond_conv(x, condition=None):
    """Concatenate conditioning vector on feature map axis."""
    if condition is None:
        return x
    else:
        reshape_shape = tf.stack([tf.shape(x)[0], 1, 1, condition.get_shape()[1]])
        out_shape = tf.stack([tf.shape(x)[0], x.get_shape()[1], x.get_shape()[2], condition.get_shape()[1]])
        to_concat = tf.reshape(condition, reshape_shape)*tf.ones(out_shape)
        return tf.concat([x, to_concat], 3)

def concat_cond_lin(tensor_in, condition):
    """Concatenate conditioning vector on feature map axis."""
    if condition is None:
        return tensor_in
    else:
        return tf.concat([tensor_in, condition, 1])

def concat_prev(tensor_in, condition):
    """Concatenate conditioning vector on feature map axis."""
    if condition is None:
        return tensor_in
    else:
        if tensor_in.get_shape()[1:3] == condition.get_shape()[1:3]:
            pad_shape = tf.stack([tf.shape(tensor_in)[0], tensor_in.get_shape()[1], tensor_in.get_shape()[2],
                                 condition.get_shape()[3]])
            return tf.concat([tensor_in, condition*tf.ones(pad_shape)], 3)
        else:
            raise ValueError('unmatched shape:', tensor_in.get_shape(), 'and', condition.get_shape())

def conv2d(tensor_in, out_channels, kernels, strides, stddev=0.02, name='conv2d', reuse=None, padding='VALID'):
    """
    Apply a 2D convolution layer on the input tensor and return the resulting tensor.

    Args:
        tensor_in (tensor): The input tensor.
        out_channels (int): The number of output channels.
        kernels (list of int): The size of the kernel. [kernel_height, kernel_width]
        strides (list of int): The stride of the sliding window. [stride_height, stride_width]
        stddev (float): The value passed to the truncated normal initializer for weights. Defaults to 0.02.
        name (str): The tenorflow variable scope. Defaults to 'conv2d'.
        reuse (bool): True to reuse weights and biases.
        padding (str): 'SAME' or 'VALID'. The type of padding algorithm to use. Defaults to 'VALID'.

    Returns:
        tensor: The resulting tensor.

    """
    if tensor_in is None:
        return None
    else:
        with tf.variable_scope(name, reuse=reuse):

            print ('|   |---'+tf.get_variable_scope().name, tf.get_variable_scope().reuse)

            weights = tf.get_variable('weights', kernels+[tensor_in.get_shape()[-1], out_channels],
                                      initializer=tf.truncated_normal_initializer(stddev=stddev))
            biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv2d(tensor_in, weights, strides=[1]+strides+[1], padding=padding)


            out_shape = tf.stack([tf.shape(tensor_in)[0]]+list(conv.get_shape()[1:]))

            return tf.reshape(tf.nn.bias_add(conv, biases), out_shape)

def transconv2d(tensor_in, out_shape, out_channels, kernels, strides, stddev=0.02, name='transconv2d', reuse=None,
                padding='VALID'):
    """
    Apply a 2D transposed convolution layer on the input tensor and return the resulting tensor.

    Args:
        tensor_in (tensor): The input tensor.
        out_shape (list of int): The output shape. [height, width]
        out_channels (int): The number of output channels.
        kernels (list of int): The size of the kernel.[kernel_height, kernel_width]
        strides (list of int): The stride of the sliding window. [stride_height, stride_width]
        stddev (float): The value passed to the truncated normal initializer for weights. Defaults to 0.02.
        name (str): The tenorflow variable scope. Defaults to 'transconv2d'.
        reuse (bool): True to reuse weights and biases.
        padding (str): 'SAME' or 'VALID'. The type of padding algorithm to use. Defaults to 'VALID'.

    Returns:
        tensor: The resulting tensor.

    """
    if tensor_in is None:
        return None
    else:
        with tf.variable_scope(name, reuse=reuse):

            print('|   |---'+tf.get_variable_scope().name, tf.get_variable_scope().reuse)

            # filter : [height, width, output_channels, in_channels]
            weights = tf.get_variable('weights', kernels+[out_channels, tensor_in.get_shape()[-1]],
                                      initializer=tf.truncated_normal_initializer(stddev=stddev))
            biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))


            output_shape =  tf.stack([tf.shape(tensor_in)[0]]+out_shape+[out_channels])

            try:
                conv_transpose = tf.nn.conv2d_transpose(tensor_in, weights, output_shape=output_shape,
                                                        strides=[1]+strides+[1], padding=padding)
            except AttributeError: # Support for verisons of TensorFlow before 0.7.0
                conv_transpose = tf.nn.deconv2d(tensor_in, weights, output_shape=output_shape, strides=[1]+strides+[1],
                                                padding=padding)

            return tf.reshape(tf.nn.bias_add(conv_transpose, biases), output_shape)

def linear(tensor_in, output_size, stddev=0.02, bias_init=0.0, name='linear', reuse=None):
    """
    Apply a linear layer on the input tensor and return the resulting tensor.

    Args:
        tensor_in (tensor): The input tensor.
        output_size (int): The output size.
        stddev (float): The value passed to the truncated normal initializer for weights. Defaults to 0.02.
        bias_init (float): The value passed to constant initializer for weights. Defaults to 0.0.
        name (str): The tenorflow variable scope. Defaults to 'linear'.
        reuse (bool): True to reuse weights and biases.
        padding (str): 'SAME' or 'VALID'. The type of padding algorithm to use. Defaults to 'VALID'.

    Returns:
        tensor: The resulting tensor.

    """
    if tensor_in is None:
        return None
    else:
        with tf.variable_scope(name, reuse=reuse):

            print ('|   |---'+tf.get_variable_scope().name, tf.get_variable_scope().reuse)

            weights = tf.get_variable('weights', [tensor_in.get_shape()[1], output_size], tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=stddev))
            biases = tf.get_variable('biases', [output_size], initializer=tf.constant_initializer(bias_init))

            lin = tf.nn.bias_add(tf.matmul(tensor_in, weights), biases)

            return lin


def to_chroma_tf(bar_or_track_bar, is_normalize=True):
    """Return the chroma tensor of the input tensor"""
    out_shape = tf.stack([tf.shape(bar_or_track_bar)[0], bar_or_track_bar.get_shape()[1], 12, 7,
                         bar_or_track_bar.get_shape()[3]])
    chroma = tf.reduce_sum(tf.reshape(tf.cast(bar_or_track_bar, tf.float32), out_shape), axis=3)
    if is_normalize:
        chroma_max = tf.reduce_max(chroma, axis=(1, 2, 3), keep_dims=True)
        chroma_min = tf.reduce_min(chroma, axis=(1, 2, 3), keep_dims=True)
        return tf.truediv(chroma - chroma_min, (chroma_max - chroma_min + 1e-15))
    else:
        return chroma

def to_binary_tf(bar_or_track_bar, threshold=0.0, track_mode=False, melody=False):
    """Return the binarize tensor of the input tensor (be careful of the channel order!)"""
    if track_mode:
        # melody track
        if melody:
            melody_is_max = tf.equal(bar_or_track_bar, tf.reduce_max(bar_or_track_bar, axis=2, keep_dims=True))
            melody_pass_threshold = (bar_or_track_bar > threshold)
            out_tensor = tf.logical_and(melody_is_max, melody_pass_threshold)
        # non-melody track
        else:
            out_tensor = (bar_or_track_bar > threshold)
        return out_tensor
    else:
        if len(bar_or_track_bar.get_shape()) == 4:
            melody_track = tf.slice(bar_or_track_bar, [0, 0, 0, 0], [-1, -1, -1, 1])
            other_tracks = tf.slice(bar_or_track_bar, [0, 0, 0, 1], [-1, -1, -1, -1])
        elif len(bar_or_track_bar.get_shape()) == 5:
            melody_track = tf.slice(bar_or_track_bar, [0, 0, 0, 0, 0], [-1, -1, -1, -1, 1])
            other_tracks = tf.slice(bar_or_track_bar, [0, 0, 0, 0, 1], [-1, -1, -1, -1, -1])
        # melody track
        melody_is_max = tf.equal(melody_track, tf.reduce_max(melody_track, axis=2, keep_dims=True))
        melody_pass_threshold = (melody_track > threshold)
        out_tensor_melody = tf.logical_and(melody_is_max, melody_pass_threshold)
        # other tracks
        out_tensor_others = (other_tracks > threshold)
        if len(bar_or_track_bar.get_shape()) == 4:
            return tf.concat([out_tensor_melody, out_tensor_others], 3)
        elif len(bar_or_track_bar.get_shape()) == 5:
            return tf.concat([out_tensor_melody, out_tensor_others], 4)


def to_image_tf(tensor_in, colormap=None):
    """Reverse the second dimension and swap the second dimension and the third dimension"""
    if colormap is None:
        colormap = get_colormap()
    shape = tf.stack([-1, tensor_in.get_shape()[1], tensor_in.get_shape()[2], 3])
    recolored = tf.reshape(tf.matmul(tf.reshape(tensor_in, [-1, 5]), colormap), shape)
    return tf.transpose(tf.reverse_v2(recolored, axis=[2]), [0, 2, 1, 3])
