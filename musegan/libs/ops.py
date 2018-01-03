import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from utils import *


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
        reshape_shape = tf.pack([tf.shape(x)[0], 1, 1, condition.get_shape()[1]])
        out_shape = tf.pack([tf.shape(x)[0], x.get_shape()[1], x.get_shape()[2], condition.get_shape()[1]])
        to_concat = tf.reshape(condition, reshape_shape)*tf.ones(out_shape)
        return tf.concat(3, [x, to_concat])

def concat_cond_lin(tensor_in, condition):
    """Concatenate conditioning vector on feature map axis."""
    if condition is None:
        return tensor_in
    else:
        return tf.concat(1, [tensor_in, condition])

def concat_prev(tensor_in, condition):
    """Concatenate conditioning vector on feature map axis."""
    if condition is None:
        return tensor_in
    else:
        if tensor_in.get_shape()[1:3] == condition.get_shape()[1:3]:
            pad_shape = tf.pack([tf.shape(tensor_in)[0], tensor_in.get_shape()[1], tensor_in.get_shape()[2],
                                 condition.get_shape()[3]])
            return tf.concat(3, [tensor_in, condition*tf.ones(pad_shape)])
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

            print '|   |---'+tf.get_variable_scope().name, tf.get_variable_scope().reuse

            weights = tf.get_variable('weights', kernels+[tensor_in.get_shape()[-1], out_channels],
                                      initializer=tf.truncated_normal_initializer(stddev=stddev))
            biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv2d(tensor_in, weights, strides=[1]+strides+[1], padding=padding)

            try:
                out_shape = tf.stack([tf.shape(tensor_in)[0]]+list(conv.get_shape()[1:]))
            except AttributeError: # Support for verisons of TensorFlow before 0.12.  0
                out_shape = tf.pack([tf.shape(tensor_in)[0]]+list(conv.get_shape()[1:]))

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

            print '|   |---'+tf.get_variable_scope().name, tf.get_variable_scope().reuse

            # filter : [height, width, output_channels, in_channels]
            weights = tf.get_variable('weights', kernels+[out_channels, tensor_in.get_shape()[-1]],
                                      initializer=tf.truncated_normal_initializer(stddev=stddev))
            biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))

            try:
                output_shape =  tf.stack([tf.shape(tensor_in)[0]]+out_shape+[out_channels])
            except AttributeError: # Support for verisons of TensorFlow before 0.12.0
                output_shape =  tf.pack([tf.shape(tensor_in)[0]]+out_shape+[out_channels])

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

            print '|   |---'+tf.get_variable_scope().name, tf.get_variable_scope().reuse

            weights = tf.get_variable('weights', [tensor_in.get_shape()[1], output_size], tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=stddev))
            biases = tf.get_variable('biases', [output_size], initializer=tf.constant_initializer(bias_init))

            lin = tf.nn.bias_add(tf.matmul(tensor_in, weights), biases)

            return lin

# def binary_cross_entropy(preds, targets, name=None):

#     eps = 1e-12

#     with ops.op_scope([preds, targets], name, 'bce_loss') as name:

#         preds = ops.convert_to_tensor(preds, name='preds')
#         targets = ops.convert_to_tensor(targets, name='targets')

#         return tf.reduce_mean(-(targets * tf.log(preds + eps) +  (1. - targets) * tf.log(1. - preds + eps)))

######################################################## Metrics #######################################################
def metric_num_pitch_used(bar_or_track_bar):
    """ binary input please """
    return tf.squeeze(tf.reduce_mean(tf.count_nonzero(tf.reduce_any(tf.cast(bar_or_track_bar, tf.bool), axis=1), axis=1,
                                                      dtype=tf.float32), axis=0))

def metric_not_in_scale(track_bar):
    """ binary input please """
    mask = tf.constant([1., 0., 1., 0., 1., 1., 0., 1., 0., 1., 0., 1.], dtype=tf.float32, name='c_scale_mask')
    mask_matrix = tf.tile(tf.expand_dims(mask, 0), [tf.shape(track_bar)[0], 1])
    track_bar_pad = tf.pad(track_bar, [[0, 0], [1, 0], [0, 0], [0, 0]], "CONSTANT")
    onsets = tf.logical_and(~track_bar_pad[:, :-1], track_bar_pad[:, 1:])
    chroma_onset = tf.squeeze(tf.reduce_sum(to_chroma_tf(onsets), axis=1), [2])
    all_notes = tf.reduce_sum(chroma_onset, axis=1)
    in_scale_notes = tf.reduce_sum(tf.multiply(chroma_onset, mask_matrix), axis=1)
    ratio = (1.0 - tf.truediv(in_scale_notes, all_notes))
    return tf.squeeze(tf.reduce_mean(ratio, axis=0))

def get_drum_filter():
    drum_filter_matrix = np.empty((6, 96), dtype=np.float32)
    drum_filter = np.tile([1., 0.3, 0., 0., 0., 0.3], 16)
    for i in range(6):
        cdf = np.roll(drum_filter, i)
        drum_filter_matrix[i, :] = cdf
    return tf.constant(drum_filter_matrix, dtype=tf.float32, name='drum_filter_matrix')

def metric_drum_pattern(track_bar, drum_filter_matrix):
    """ binary input """
    temporal = tf.reduce_sum(tf.cast(track_bar, tf.float32), axis=2)
    all_notes = tf.reduce_sum(tf.squeeze(tf.cast(temporal, tf.float32), [2]), axis=1, keep_dims=True)
    temporal_tile = tf.tile(temporal, [1, 1, 6])
    df_matrix = tf.transpose(tf.expand_dims(drum_filter_matrix, 0), [0, 2, 1])
    df_matrix_tile = tf.tile(df_matrix, [tf.shape(track_bar)[0], 1, 1])
    score = tf.reduce_sum(tf.multiply(df_matrix_tile, temporal_tile), 1)
    max_score = tf.reduce_max(score, axis=1, keep_dims=True)
    return tf.squeeze(tf.reduce_mean(tf.truediv(max_score, all_notes), axis=0))

def metric_polyphonic_ratio(track_bar, threshold=3):
    """ binary input """
    polyphonic_track_bar = (tf.count_nonzero(track_bar, axis=2) >= threshold)
    return tf.squeeze(tf.reduce_mean(tf.count_nonzero(polyphonic_track_bar, axis=1, dtype=tf.float32)/96.0, axis=0))

def metric_too_short_note_ratio(track_bar, threshold=2):
    """ binary input """
    track_bar_pad = tf.pad(track_bar, [[0, 0], [1, 1], [0, 0], [0, 0]], "CONSTANT")
    onsets = tf.where(tf.transpose(tf.logical_and(~track_bar_pad[:, :-2], track_bar_pad[:, 1:-1]), [0, 2, 1, 3]))
    offsets = tf.where(tf.transpose(tf.logical_and(track_bar_pad[:, 1:-1], ~track_bar_pad[:, 2:]), [0, 2, 1, 3]))
    num_unqualified_note = tf.count_nonzero((offsets[:, 2]-onsets[:, 2] <= threshold), dtype=tf.int32)
    return tf.squeeze(tf.truediv(num_unqualified_note, tf.shape(onsets)[0]))

def get_tonal_transform_matrix(r1=1.0, r2=1.0, r3=0.5):
    tt_matrix = np.empty((6, 12), dtype=np.float32)
    tt_matrix[0, :] = r1*np.sin(np.arange(12)*(7./6.)*np.pi)
    tt_matrix[1, :] = r1*np.cos(np.arange(12)*(7./6.)*np.pi)
    tt_matrix[2, :] = r2*np.sin(np.arange(12)*(3./2.)*np.pi)
    tt_matrix[3, :] = r2*np.cos(np.arange(12)*(3./2.)*np.pi)
    tt_matrix[4, :] = r3*np.sin(np.arange(12)*(2./3.)*np.pi)
    tt_matrix[5, :] = r3*np.cos(np.arange(12)*(2./3.)*np.pi)
    return tf.constant(tt_matrix, dtype=tf.float32, name='tonal_transform_matrix')

def metric_harmonicity(bar, tt_matrix=None):
    """ chroma input please """
    melody_rhythm = tf.slice(bar, [0, 0, 0, 0], [-1, -1, -1, 2])
    out_shape = tf.pack([tf.shape(melody_rhythm)[0], 4, 24, melody_rhythm.get_shape()[2], melody_rhythm.get_shape()[3]])
    beat_chroma = tf.reduce_sum(tf.reshape(melody_rhythm, out_shape), axis=2)
    beat_chroma = tf.truediv(beat_chroma, tf.reduce_sum(beat_chroma, axis=2, keep_dims=True) + 1e-15)
    tonal_centroid = tf.matmul(tt_matrix, tf.reshape(tf.transpose(beat_chroma, [2, 0, 1, 3]), [12, -1]))
    out_shape = tf.pack([6, tf.shape(melody_rhythm)[0], 4, melody_rhythm.get_shape()[3]])
    tonal_centroid = tf.transpose(tf.reshape(tonal_centroid, out_shape), [1, 2, 0, 3])
    tonal_distance = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.slice(tonal_centroid, [0, 0, 0, 0], \
                     [-1, -1, -1, 1]) - tf.slice(tonal_centroid, [0, 0, 0, 1], [-1, -1, -1, 1])), axis=2)), axis=1)
    return tf.squeeze(tf.reduce_mean(tonal_distance, axis=0))

######################################################## Others ########################################################
def to_chroma_tf(bar_or_track_bar, is_normalize=True):
    """Return the chroma tensor of the input tensor"""
    out_shape = tf.pack([tf.shape(bar_or_track_bar)[0], bar_or_track_bar.get_shape()[1], 12, 7,
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
            return tf.concat(3, [out_tensor_melody, out_tensor_others])
        elif len(bar_or_track_bar.get_shape()) == 5:
            return tf.concat(4, [out_tensor_melody, out_tensor_others])


def to_image_tf(tensor_in, colormap=None):
    """Reverse the second dimension and swap the second dimension and the third dimension"""
    if colormap is None:
        colormap = get_colormap()
    shape = tf.pack([-1, tensor_in.get_shape()[1], tensor_in.get_shape()[2], 3])
    recolored = tf.reshape(tf.matmul(tf.reshape(tensor_in, [-1, 5]), colormap), shape)
    return tf.transpose(tf.reverse_v2(recolored, axis=[2]), [0, 2, 1, 3])

    # recolored_bars = np.zeros((bars.shape[0], bars.shape[1], bars.shape[2], 3))
    # for track_idx in range(bars.shape[-1]):
    #     recolored_bars = recolored_bars + bars[..., track_idx][:, :, :, None]*colormap[track_idx][None, None, None, :]
    # return np.flip(np.transpose(recolored_bars, (0, 2, 1, 3)), axis=1)
