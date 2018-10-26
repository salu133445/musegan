"""Tensorflow ops."""
import tensorflow as tf

CONV_KERNEL_INITIALIZER = tf.truncated_normal_initializer(stddev=0.05)
DENSE_KERNEL_INITIALIZER = tf.truncated_normal_initializer(stddev=0.05)

dense = lambda i, u: tf.layers.dense(
    i, u, kernel_initializer=DENSE_KERNEL_INITIALIZER)
conv2d = lambda i, f, k, s: tf.layers.conv2d(
    i, f, k, s, kernel_initializer=CONV_KERNEL_INITIALIZER)
conv3d = lambda i, f, k, s: tf.layers.conv3d(
    i, f, k, s, kernel_initializer=CONV_KERNEL_INITIALIZER)
tconv2d = lambda i, f, k, s: tf.layers.conv2d_transpose(
    i, f, k, s, kernel_initializer=CONV_KERNEL_INITIALIZER)
tconv3d = lambda i, f, k, s: tf.layers.conv3d_transpose(
    i, f, k, s, kernel_initializer=CONV_KERNEL_INITIALIZER)

def get_normalization(norm_type, training=None):
    """Return the normalization function."""
    if norm_type == 'batch_norm':
        return lambda x: tf.layers.batch_normalization(x, training=training)
    if norm_type == 'layer_norm':
        return tf.contrib.layers.layer_norm
    if norm_type is None or norm_type == '':
        return lambda x: x
    raise ValueError("Unrecognizable normalization type.")
