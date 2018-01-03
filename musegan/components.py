import tensorflow as tf
import numpy as np
from libs.ops import *
from libs.utils import *
from modules import *

class Nowbar(object):
    def __init__(self, config):
        with tf.variable_scope('NowBar'):
            self.z_intra = tf.placeholder(tf.float32, shape=[None, config.z_intra_dim, config.track_dim], name='z_intra')
            self.z_inter = tf.placeholder(tf.float32, shape=[None, config.z_inter_dim], name='z_inter')
            self.x = tf.placeholder(tf.float32, shape=[None, config.output_w, config.output_h, config.track_dim], name='x')

            # set accompany track
            if config.acc_idx  is not None:
                self.acc_track = tf.slice(self.x, [0, 0, 0, config.acc_idx], [-1, -1, -1, 1]) # take piano as condition
                BE = BarEncoder()
                self.nowbar = BE(self.acc_track)
            else:
                self.nowbar = None
                self.out_track_dim = config.track_dim

            self._build_generator(config)
            self._build_discriminator(config)
            self._build_optimizer(config)

            with tf.variable_scope('summary/d'):
                self.summary_d_real = tf.summary.histogram('d/d_real', self.D_real)
                self.summary_d_fake = tf.summary.histogram('d/d_fake', self.D_fake)

            with tf.variable_scope('summary/g'):
                self.summary_g_loss_hist = tf.summary.histogram('g_loss', self.g_loss)
                self.summary_g_loss_scalar = tf.summary.scalar('g_loss', self.g_loss)

            with tf.variable_scope('summary/d'):
                self.summary_d_loss_hist = tf.summary.histogram('d_loss', self.d_loss)
                self.summary_d_loss_scalar = tf.summary.scalar('d_loss', self.d_loss)

            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
            self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=tf.get_variable_scope().name)
            self.summary = tf.summary.merge(self.summaries)
            self.summary_d = tf.summary.merge([s for s in self.summaries if '/d/' in s.name])
            self.summary_g = tf.summary.merge([s for s in self.summaries if '/g/' in s.name])

    def _build_generator(self, config):
        with tf.variable_scope('G') as scope:
            self.all_tracks = []

            for tidx in range(config.track_dim):
                if tidx is config.acc_idx:
                    tmp_track = self.acc_track
                else:
                    with tf.variable_scope(config.track_names[tidx]):
                        z_intra =  tf.squeeze(tf.slice(self.z_intra, [0, 0, tidx], [-1, -1, 1]), squeeze_dims=2)
                        latent_track = tf.concat(1, [z_intra, self.z_inter])
                        BG = BarGenerator(name='BG')
                        tmp_track = BG(in_tensor=latent_track, nowbar=self.nowbar)

                self.all_tracks.append(tmp_track)

            self.prediction = tf.concat(3, [t for t in self.all_tracks])
            self.prediction_binary = to_binary_tf(self.prediction)
            self.prediction_chroma = to_chroma_tf(self.prediction_binary)

            self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

            ## susmary
            prediction_image = to_image_tf(self.prediction, config.colormap)
            self.summary_prediction_image = tf.summary.image('prediction/G', prediction_image,
                                                             max_outputs=10)


    def _build_discriminator(self, config):
        with tf.variable_scope('D') as scope:
            BD = BarDiscriminator()

            self.input_real = self.x
            self.input_fake = self.prediction

            _, self.D_real = BD(self.input_real, reuse=False)
            _, self.D_fake = BD(self.input_fake, reuse=True)

            epsilon = tf.random_uniform([], 0.0, 1.0)

            X_hat = epsilon * self.input_real + (1 - epsilon) * self.input_fake
            _, D_hat = BD(X_hat, reuse=True)

            self.d_loss = tf.reduce_mean(self.D_real) - tf.reduce_mean(self.D_fake)
            self.g_loss = tf.reduce_mean(self.D_fake)

            gp = tf.gradients(D_hat, X_hat)[0]
            gp = tf.sqrt(tf.reduce_sum(tf.square(gp), axis=1))
            gp = tf.reduce_mean(tf.square(gp - 1.0) * config.lamda)

            self.d_loss += gp
            self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

    def _build_optimizer(self, config):
         with tf.variable_scope('Opt'):

            self.d_optim = tf.train.AdamOptimizer(config.lr, beta1=config.beta1, beta2=config.beta2) \
                                   .minimize(self.d_loss, var_list=self.d_vars)

            self.g_optim = tf.train.AdamOptimizer(config.lr, beta1=config.beta1, beta2=config.beta2) \
                                   .minimize(self.g_loss, var_list=self.g_vars)

    def sample():
        pass

def model_config(config):
    if config.__name__ is 'NowBarConfig':
        Model = Nowbar(config)
    elif config.__name__ is 'NowBarConfig':
        Model = Temporal(config)
    else:
        raise NameError('Unknow config')
    return Model
