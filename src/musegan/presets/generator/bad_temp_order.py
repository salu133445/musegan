"""This file defines the network architecture for the generator."""
import tensorflow as tf
from tensorflow.nn import relu, leaky_relu, tanh, sigmoid
from ..ops import tconv3d, get_normalization

NORMALIZATION = 'batch_norm' # 'batch_norm', 'layer_norm'
ACTIVATION = relu # relu, leaky_relu, tanh, sigmoid

class Generator:
    def __init__(self, n_tracks, name='Generator'):
        self.n_tracks = n_tracks
        self.name = name

    def __call__(self, tensor_in, condition=None, training=None, slope=None):
        norm = get_normalization(NORMALIZATION, training)
        tconv_layer = lambda i, f, k, s: ACTIVATION(norm(tconv3d(i, f, k, s)))

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            h = tensor_in
            h = tf.expand_dims(tf.expand_dims(tf.expand_dims(h, 1), 1), 1)

            # Shared network
            with tf.variable_scope('shared'):
                h = tconv_layer(h, 512, (4, 1, 1), (4, 1, 1))        # 4, 1, 1
                h = tconv_layer(h, 256, (1, 3, 3), (1, 3, 3))        # 4, 3, 3
                h = tconv_layer(h, 128, (1, 4, 3), (1, 4, 2))        # 4, 12, 7

            # Pitch-time private network
            with tf.variable_scope('pitch_time_private'):
                s1 = [tconv_layer(h, 32, (1, 1, 12), (1, 1, 12))     # 4, 12, 12
                      for _ in range(self.n_tracks)]
                s1 = [tconv_layer(s1[i], 16, (1, 4, 1), (1, 4, 1))   # 4, 48, 84
                      for i in range(self.n_tracks)]

            # Time-pitch private network
            with tf.variable_scope('time_pitch_private'):
                s2 = [tconv_layer(h, 32, (1, 4, 1), (1, 4, 1))       # 4, 48, 7
                      for _ in range(self.n_tracks)]
                s2 = [tconv_layer(s2[i], 16, (1, 1, 12), (1, 1, 12)) # 4, 48, 84
                      for i in range(self.n_tracks)]

            h = [tf.concat((s1[i], s2[i]), -1) for i in range(self.n_tracks)]

            # Merged private network
            with tf.variable_scope('merged_private'):
                h = [norm(tconv3d(h[i], 1, (1, 1, 1), (1, 1, 1)))    # 4, 48, 84
                     for i in range(self.n_tracks)]
                h = tf.concat(h, -1)

        return tanh(h)
