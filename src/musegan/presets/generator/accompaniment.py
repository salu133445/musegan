"""This file defines the network architecture for the generator."""
import tensorflow as tf
from tensorflow.nn import relu, leaky_relu, tanh, sigmoid
from ..ops import conv3d, tconv3d, get_normalization

NORMALIZATION = 'batch_norm' # 'batch_norm', 'layer_norm'
ACTIVATION = relu # relu, leaky_relu, tanh, sigmoid

class Generator:
    def __init__(self, n_tracks, condition_track_idx=None, name='Generator'):
        self.n_tracks = n_tracks
        self.condition_track_idx = condition_track_idx
        self.name = name

    def __call__(self, tensor_in, condition=None, condition_track=None,
                 training=None, slope=None):

        norm = get_normalization(NORMALIZATION, training)
        conv_layer = lambda i, f, k, s: ACTIVATION(norm(conv3d(i, f, k, s)))
        tconv_layer = lambda i, c, f, k, s: ACTIVATION(norm(tconv3d(
            tf.concat((i, c), -1), f, k, s)))

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            # ---------------------------- Encoder -----------------------------
            c = condition_track

            # Pitch-time private network
            with tf.variable_scope('encoder_pitch_time'):
                pt_1 = conv_layer(c, 16, (1, 1, 12), (1, 1, 12))     # 4, 48, 7
                pt_2 = conv_layer(pt_1, 32, (1, 3, 1), (1, 3, 1))    # 4, 16, 7

            # Time-pitch private network
            with tf.variable_scope('encoder_time_pitch'):
                tp_1 = conv_layer(c, 16, (1, 3, 1), (1, 3, 1))       # 4, 16, 84
                tp_2 = conv_layer(tp_1, 32, (1, 1, 12), (1, 1, 12))  # 4, 16, 7

            shared = tf.concat((tp_2, pt_2), -1)

            # Shared network
            with tf.variable_scope('encoder_shared'):
                s1 = conv_layer(shared, 64, (1, 4, 3), (1, 4, 2))    # 4, 4, 3
                s2 = conv_layer(s1, 128, (1, 4, 3), (1, 4, 3))       # 4, 1, 1
                s3 = conv_layer(s2, 256, (4, 1, 1), (4, 1, 1))       # 1, 1, 1

            # --------------------------- Generator ----------------------------
            h = tensor_in
            h = tf.expand_dims(tf.expand_dims(tf.expand_dims(h, 1), 1), 1)

            # Shared network
            with tf.variable_scope('shared'):
                h = tconv_layer(h, s3, 512, (4, 1, 1), (4, 1, 1))    # 4, 1, 1
                h = tconv_layer(h, s2, 256, (1, 4, 3), (1, 4, 3))    # 4, 4, 3
                h = tconv_layer(h, s1, 128, (1, 4, 3), (1, 4, 2))    # 4, 16, 7

            # Pitch-time private network
            with tf.variable_scope('pitch_time_private'):
                s1 = [tconv_layer(h, tp_2, 32, (1, 1, 12), (1, 1, 12))
                      for _ in range(self.n_tracks)]                 # 4, 16, 84
                s1 = [tconv_layer(s1[i], tp_1, 16, (1, 3, 1), (1, 3, 1))
                      for i in range(self.n_tracks)]                 # 4, 48, 84

            # Time-pitch private network
            with tf.variable_scope('time_pitch_private'):
                s2 = [tconv_layer(h, pt_2, 32, (1, 3, 1), (1, 3, 1))
                      for _ in range(self.n_tracks)]                 # 4, 48, 7
                s2 = [tconv_layer(s2[i], pt_1, 16, (1, 1, 12), (1, 1, 12))
                      for i in range(self.n_tracks)]                 # 4, 48, 84

            h = [tf.concat((s1[i], s2[i]), -1) for i in range(self.n_tracks)]

            # Merged private network
            with tf.variable_scope('merged_private'):
                h = [tanh(norm(tconv3d(h[i], 1, (1, 1, 1), (1, 1, 1))))
                     for i in range(self.n_tracks)]                  # 4, 48, 84

                return tf.concat((
                    h[:self.condition_track_idx] + [c] +
                    h[self.condition_track_idx:]), -1)
