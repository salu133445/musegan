from __future__ import print_function
import tensorflow as tf
import numpy as np
from libs.ops import *
from libs.utils import *

class Generator(object):
    def __init__(self, name, output_dim, is_bn):
        self.name = name
        self.is_bn = is_bn

    def get_trainable_variables(self):
        t_vars = tf.trainable_variables()
        t_vars_model = {v.name: v for v in t_vars if self.name in v.name}
        return t_vars_model

class PhraseGenerator(object):
    def __init__(self, name='PG', output_dim=1, is_bn=True):
        self.output_dim = output_dim
        self.name = name
        self.is_bn = is_bn

    def __call__(self, in_tensor, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            h0 = tf.reshape(in_tensor, tf.pack([-1, 1, 1, in_tensor.get_shape()[1]]))
            h0 = relu(batch_norm(transconv2d(concat_cond_conv(h0, self.condition), [2, 1], 1024, kernels=[2, 1],
                                            strides=[2, 1], name='h1'), self.bn))
            h1 = relu(batch_norm(transconv2d(concat_cond_conv(h0, self.condition), [4, 1], self.output_dim,
                                            kernels=[3, 1], strides=[1, 1], name='h2'), self.bn))
            h1 = tf.transpose(tf.squeeze(h1, axis=2), [0, 2, 1])

        return h1

class BarGenerator(object):
    def __init__(self, name='BG', output_dim=1, is_bn=True):
        self.output_dim = output_dim
        self.name = name
        self.is_bn = is_bn

    def __call__(self, in_tensor, nowbar, reuse=False):

        with tf.variable_scope(self.name, reuse=reuse):

            h0 = tf.reshape(in_tensor, tf.pack([-1, 1, 1, in_tensor.get_shape()[1]]))
            h0 = relu(batch_norm(transconv2d(h0, [1, 1], 1024, kernels=[1, 1], strides=[1, 1], name='h0'), self.is_bn))

            h1 = tf.reshape(h0, [-1, 2, 1, 512])
            h1 = concat_prev(h1, nowbar[6])
            h1 = relu(batch_norm(transconv2d(h1, [4, 1], 512, kernels=[2, 1], strides=[2, 1], name='h1'), self.is_bn))

            h2 = concat_prev(h1, nowbar[5])
            h2 = relu(batch_norm(transconv2d(h2, [8, 1], 256, kernels=[2, 1], strides=[2, 1], name='h2'), self.is_bn))

            h3 = concat_prev(h2, nowbar[4])
            h3 = relu(batch_norm(transconv2d(h3, [16, 1], 256, kernels=[2, 1], strides=[2, 1], name='h3'), self.is_bn))

            h4 = concat_prev(h3, nowbar[3])
            h4 = relu(batch_norm(transconv2d(h4, [32, 1], 128, kernels=[2, 1], strides=[2, 1], name='h4'), self.is_bn))

            h5 = concat_prev(h4, nowbar[2])
            h5 = relu(batch_norm(transconv2d(h5, [96, 1], 128, kernels=[3, 1], strides=[3, 1], name='h5'), self.is_bn))

            h6 = concat_prev(h5, nowbar[1])
            h6 = relu(batch_norm(transconv2d(h6, [96, 7], 64, kernels=[1, 7], strides=[1, 1], name='h6'), self.is_bn))

            h7 = concat_prev(h6, nowbar[0])
            h7 = transconv2d(h7, [96, 84], self.output_dim, kernels=[1, 12], strides=[1, 12], name='h7')

        return tf.nn.tanh(h7)

class BarEncoder(object):
    def __init__(self, name='BE', is_bn=True):
        self.name = name
        self.is_bn = is_bn

    def __call__(self, in_tensor, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            h0 = lrelu(batch_norm(conv2d(in_tensor, 16, kernels=[1, 12], strides=[1, 12], name='h0'), self.is_bn))
            h1 = lrelu(batch_norm(conv2d(h0, 16, kernels=[1, 7], strides=[1, 7], name='h1'), self.is_bn))
            h2 = lrelu(batch_norm(conv2d(h1, 16, kernels=[3, 1], strides=[3, 1], name='h2'), self.is_bn))
            h3 = lrelu(batch_norm(conv2d(h2, 16, kernels=[2, 1], strides=[2, 1], name='h3'), self.is_bn))
            h4 = lrelu(batch_norm(conv2d(h3, 16, kernels=[2, 1], strides=[2, 1], name='h4'), self.is_bn))
            h5 = lrelu(batch_norm(conv2d(h4, 16, kernels=[2, 1], strides=[2, 1], name='h5'), self.is_bn))
            h6 = lrelu(batch_norm(conv2d(h5, 16, kernels=[2, 1], strides=[2, 1], name='h6'), self.is_bn))

            return [h0, h1, h2, h3, h4, h5, h6]

class BarDiscriminator(object):

    def __init__(self, name='BD'):
        self.name = name

    def __call__(self, in_tensor, reuse):
        with tf.variable_scope(self.name, reuse=reuse):

            ## conv
            h0 = lrelu(conv2d(in_tensor, 128, kernels=[1, 12], strides=[1, 12], name='h0'))
            h1 = lrelu(conv2d(h0, 128, kernels=[1, 7], strides=[1, 7], name='h1'))
            h2 = lrelu(conv2d(h1, 128, kernels=[2, 1], strides=[2, 1], name='h2'))
            h3 = lrelu(conv2d(h2, 128, kernels=[2, 1], strides=[2, 1], name='h3'))
            h4 = lrelu(conv2d(h3, 256, kernels=[4, 1], strides=[2, 1], name='h4'))
            h5 = lrelu(conv2d(h4, 512, kernels=[3, 1], strides=[2, 1], name='h5'))

            ## linear
            h6 = tf.reshape(h5, [-1, np.product([s.value for s in h5.get_shape()[1:]])])
            h6 = lrelu(linear(h6, 1024, name='h6'))
            h7 = linear(h6, 1, name='h7')
            return h5, h7

class PhraseDiscriminator(object):

    def __init__(self, name='BD'):
        self.name = name

    def __call__(self, in_tensor, reuse):
        with tf.variable_scope(self.name):
            h0 = lrelu(conv2d(concat_cond_conv(tf.expand_dims(in_tensor, axis=2), self.condition), 512,
                              kernels=[2, 1], strides=[1, 1], name='h0'))
            h1 = lrelu(conv2d(concat_cond_conv(h0, self.condition), self.output_dim, kernels=[3, 1], strides=[3, 1],
                              name='h1'))
            h1 = tf.squeeze(tf.squeeze(h1, axis=2), axis=1)
        return h1
