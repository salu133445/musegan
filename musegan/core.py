from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from glob import glob
import numpy as np
import tensorflow as tf
from six.moves import xrange
from sklearn.utils import shuffle
from musegan.libs.ops import *
from musegan.libs.utils import *
from musegan.eval.metrics import *
from musegan.components import *
from config import *
# from config.default_128_r_off_y_off import *

###########################################################################
# GAN
###########################################################################

class GAN(object):
    def __init__(self, sess, config, model):
        print ('{:=^120}'.format(' Building MidiNet '))

        self.config = config
        self.sess = sess

        # create global step variable and increment op
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.global_step_increment = tf.assign(self.global_step, self.global_step+1)

        # create generator (G)
        self.model = model

        self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        self.summary = tf.summary.merge(self.summaries)
        self.summary_image = tf.summary.merge([s for s in self.summaries if '/prediction/' in s.name])

        self.model.get_model_info(quiet=False)

        """ Saver """
        self.saver = tf.train.Saver()
        self.saver_g = tf.train.Saver(self.model.g_vars, max_to_keep=30)
        self.saver_d = tf.train.Saver(self.model.d_vars, max_to_keep=30)
        self.saver_dict = {'midinet': self.saver, 'G':self.saver_g, 'D':self.saver_d}
        print( '{:=^120}'.format('Done!'))

        print('*initializing variables...')

        tf.global_variables_initializer().run()

        self.dir_ckpt = os.path.join(self.config.exp_name, 'checkpoint')
        self.dir_sample = os.path.join(self.config.exp_name, 'samples')
        self.dir_log = os.path.join(self.config.exp_name, 'logs')

        if not os.path.exists(self.dir_ckpt):
            os.makedirs(self.dir_ckpt)
        if not os.path.exists(self.dir_sample):
            os.makedirs(self.dir_sample)
        if not os.path.exists(self.dir_log):
            os.makedirs(self.dir_log)

        path_src = os.path.join(self.dir_log, 'src')

        if not os.path.exists(path_src):
            os.makedirs(path_src)

        for file_path in glob.glob("./*.py"):
            copyfile(file_path, os.path.join(path_src, os.path.basename(file_path)))

    def train(self, input_data):

        # save training data samples
        sample_shape = get_sample_shape(self.config.sample_size)
        imsave(input_data.get_rand_smaples(self.config.sample_size), sample_shape, path=os.path.join(self.dir_sample, 'Train(random).png'))

        feed_dict_sample = input_data.gen_feed_dict()

        # training
        counter = 0
        num_batch = input_data.get_batch_num()

        for epoch in range(self.config.epoch):

            print ('{:-^120}'.format(' Epoch {} Start '.format(epoch)))
            epoch_start_time = time.time()

            for batch_idx in range(num_batch):

                batch_start_time = time.time()

                feed_dict_batch = input_data.gen_feed_dict(idx=batch_idx)

                # update D
                num_iters_D = 100 if counter < 25 or counter % 500 == 0 else 5

                for j in range(num_iters_D):
                    self.sess.run(self.model.d_optim, feed_dict=feed_dict_batch)

                # update G
                self.sess.run(self.model.g_optim, feed_dict=feed_dict_batch)

                # compute losses
                d_loss_eval = self.model.d_loss.eval(feed_dict_batch) * -1.
                g_loss_eval = self.model.g_loss.eval(feed_dict_batch)


                # print and save batch info
                if self.config.print_batch:
                    print( '---{}--- epoch: {:2d} | batch: {:4d}/{:4d} | time: {:6.2f} '\
                        .format(self.config.exp_name+'_GPU_'+self.config.gpu_num, epoch,
                        batch_idx , num_batch, time.time() - batch_start_time))

                    print ('D loss: %6.2f, G loss: %6.2f' % (d_loss_eval, g_loss_eval))

                if not counter % self.config.iter_to_save:
                    self.run_sampler(feed_dict=feed_dict_sample,
                                        sample_size=self.config.sample_size,
                                        prefix='train_{:03d}'.format(counter),
                                        save_info=True,
                                        save_dir=self.dir_sample)

                    self.save(self.dir_ckpt, component='GD', global_step=self.global_step)

                counter += 1
                self.sess.run(self.global_step_increment)

        print ('{:=^120}'.format(' Training End '))

    def save(self, checkpoint_dir, component='all', global_step=None):

        if component == 'all':
            saver_names = ['midinet', 'G', 'D', 'invG']
        elif component == 'GD':
            saver_names = ['midinet', 'G', 'D']
        elif component == 'invG':
            saver_names = ['midinet', 'invG']

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        print('*saving checkpoints...')
        for saver_name, saver in self.saver_dict.iteritems():
            if saver_name in saver_names:
                if not os.path.exists(os.path.join(checkpoint_dir, saver_name)):
                    os.makedirs(os.path.join(checkpoint_dir, saver_name))
                saver.save(self.sess, os.path.join(checkpoint_dir, saver_name, saver_name),
                           global_step=global_step)

    def load(self, checkpoint_dir, component='all'):

        if component == 'all':
            saver_names = ['midinet'] # ['midinet', 'G', 'D', 'invG']
        elif component == 'GD':
            saver_names = ['G', 'D']

        print('*reading checkpoints...')

        for saver_name in saver_names:
            ckpt = tf.train.get_checkpoint_state(os.path.join(checkpoint_dir, saver_name))
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, saver_name, ckpt_name))
                print('*load chekpoints sucessfully')
                return True
            else:
                print('[!] Load failed...')
                return False

    def run_sampler(self, feed_dict, sample_size, prefix='sample', save_info=True, save_dir='./'):

        # run sampler
        print ('*running sampler...')
        samples = self.sess.run(self.model.prediction, feed_dict=feed_dict)
        # save results to image files
        if save_info:
            print('*saving files...')

            sample_shape = get_sample_shape(sample_size)
            imsave(samples, size=sample_shape, path=os.path.join(save_dir, prefix+'.png'))

        return samples

    def gen_test(self, input_data, gen_dir=None, batch_size=None, num_batch=1, key='test', is_eval=True, is_save=True):
        '''
            type_ = 'npy', 'data', 'all'
        '''
        batch_size = self.config.batch_size if not batch_size else batch_size

        gen_dir = os.path.join(self.config.exp_name, 'gen') if gen_dir is None else gen_dir

        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)

        prediction_list = []

        for bidx in range(num_batch):
            feed_dict = input_data.gen_feed_dict(idx=bidx, data_size=batch_size, key=key)
            s = self.run_sampler(feed_dict=feed_dict,
                                        sample_size=batch_size,
                                        save_info=is_save,
                                        save_dir=gen_dir)
            prediction_list.append(s)

        result = np.concatenate(prediction_list, axis=0)

        if is_save:
            np.save(os.path.join(gen_dir, 'gen.npy'), result)

        return result, eval_result

###########################################################################
# MuseGAN
###########################################################################

class MuseGAN(object):
    def __init__(self, sess, config, model):
        print ('{:=^120}'.format(' Building MidiNet '))


        self.config = config
        self.sess = sess

        # create global step variable and increment op
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.global_step_increment = tf.assign(self.global_step, self.global_step+1)

        # create generator (G)
        self.model = model

        self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        self.summary = tf.summary.merge(self.summaries)
        self.summary_image = tf.summary.merge([s for s in self.summaries if '/prediction/' in s.name])

        self.model.get_model_info(quiet=False)

        """ Saver """
        self.saver = tf.train.Saver()
        self.saver_g = tf.train.Saver(self.model.g_vars, max_to_keep=30)
        self.saver_d = tf.train.Saver(self.model.d_vars, max_to_keep=30)
        self.saver_dict = {'midinet': self.saver, 'G':self.saver_g, 'D':self.saver_d}
        print( '{:=^120}'.format('Done!'))

        print('*initializing variables...')



        # init metrics amd loss collection
        self.metrics = Metrics(eval_map=self.config.eval_map,
                    inter_pair=self.config.inter_pair,
                    drum_filter=self.config.drum_filter,
                    scale_mask=self.config.scale_mask,
                    track_names=self.config.track_names)

        tf.global_variables_initializer().run()

        self.dir_ckpt = os.path.join(self.config.exp_name, 'checkpoint')
        self.dir_sample = os.path.join(self.config.exp_name, 'samples')
        self.dir_log = os.path.join(self.config.exp_name, 'logs')

        if not os.path.exists(self.dir_ckpt):
            os.makedirs(self.dir_ckpt)
        if not os.path.exists(self.dir_sample):
            os.makedirs(self.dir_sample)
        if not os.path.exists(self.dir_log):
            os.makedirs(self.dir_log)

        path_src = os.path.join(self.dir_log, 'src')

        if not os.path.exists(path_src):
            os.makedirs(path_src)

        for file_path in glob.glob("./*.py"):
            copyfile(file_path, os.path.join(path_src, os.path.basename(file_path)))



    def train(self, input_data):

        # save training data samples
        sample_shape = get_sample_shape(self.config.sample_size)
        train_samples = input_data.get_rand_smaples(self.config.sample_size)
        save_bars(train_samples, sample_shape, file_path=self.dir_sample, name='Train(random).png')
        # save_midis(train_samples, file_path=os.path.join(self.dir_sample, 'train.mid'))

        if self.config.is_eval:
            #evaluation
            self.metrics.eval(input_data.x['train'][:256], quiet=False)

        feed_dict_sample = input_data.gen_feed_dict()

        # training
        counter = 0
        num_batch = input_data.get_batch_num()

        for epoch in range(self.config.epoch):

            print ('{:-^120}'.format(' Epoch {} Start '.format(epoch)))
            epoch_start_time = time.time()

            for batch_idx in range(num_batch):

                batch_start_time = time.time()

                feed_dict_batch = input_data.gen_feed_dict(idx=batch_idx)

                # update D
                num_iters_D = 100 if counter < 25 or counter % 500 == 0 else 5

                for j in range(num_iters_D):
                    self.sess.run(self.model.d_optim, feed_dict=feed_dict_batch)

                # update G
                self.sess.run(self.model.g_optim, feed_dict=feed_dict_batch)

                # compute losses
                d_loss_eval = self.model.d_loss.eval(feed_dict_batch) * -1.
                g_loss_eval = self.model.g_loss.eval(feed_dict_batch)

                # evaluation
                if self.config.is_eval:
                    samples_binary = self.sess.run(self.model.prediction_binary, feed_dict=feed_dict_batch)
                    print(samples_binary.shape)
                    score_matrix, score_pair = self.metrics.eval(samples_binary, quiet=True)
                    self.metrics.collect(score_matrix, score_pair)

                # collect loss
                self.metrics.collect_loss({'d':d_loss_eval, 'g':g_loss_eval}) # loss

                # print and save batch info
                if self.config.print_batch:
                    print( '---{}--- epoch: {:2d} | batch: {:4d}/{:4d} | time: {:6.2f} '\
                        .format(self.config.exp_name+'_GPU_'+self.config.gpu_num, epoch,
                        batch_idx , num_batch, time.time() - batch_start_time))

                    print ('D loss: %6.2f, G loss: %6.2f' % (d_loss_eval, g_loss_eval))

                if not counter % self.config.iter_to_save:
                    if self.config.is_eval:
                        self.metrics.eval(samples_binary, quiet=False)

                    self.metrics.save_history(self.dir_log)
                    self.run_sampler(feed_dict=feed_dict_sample,
                                        sample_size=self.config.sample_size,
                                        prefix='train_{:03d}'.format(counter),
                                        save_info=True,
                                        save_dir=self.dir_sample)

                    self.save(self.dir_ckpt, component='GD', global_step=self.global_step)

                counter += 1
                self.sess.run(self.global_step_increment)

        print ('{:=^120}'.format(' Training End '))

    def save(self, checkpoint_dir, component='all', global_step=None):

        if component == 'all':
            saver_names = ['midinet', 'G', 'D', 'invG']
        elif component == 'GD':
            saver_names = ['midinet', 'G', 'D']
        elif component == 'invG':
            saver_names = ['midinet', 'invG']

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        print('*saving checkpoints...')
        # for saver_name, saver in self.saver_dict.iteritems():
        for saver_name in self.saver_dict.keys():
            saver = self.saver_dict[saver_name]
            if saver_name in saver_names:
                if not os.path.exists(os.path.join(checkpoint_dir, saver_name)):
                    os.makedirs(os.path.join(checkpoint_dir, saver_name))
                saver.save(self.sess, os.path.join(checkpoint_dir, saver_name, saver_name),
                           global_step=global_step)

    def load(self, checkpoint_dir, component='all'):

        if component == 'all':
            saver_names = ['midinet'] # ['midinet', 'G', 'D', 'invG']
        elif component == 'GD':
            saver_names = ['G', 'D']

        print('*reading checkpoints...')

        for saver_name in saver_names:
            ckpt = tf.train.get_checkpoint_state(os.path.join(checkpoint_dir, saver_name))
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, saver_name, ckpt_name))
                print('*load chekpoints sucessfully')
                return True
            else:
                print('[!] Load failed...')
                return False

    def run_sampler(self, feed_dict, sample_size, prefix='sample', save_info=True, save_dir='./',type_=0):

        # run sampler
        print ('*running sampler...')
        samples, samples_binary, samples_chroma = self.sess.run([self.model.prediction, self.model.prediction_binary,
                                                                self.model.prediction_chroma], feed_dict=feed_dict)

        # save results to image files
        if save_info:
            print('*saving files...')
            save_midis(samples_binary, file_path=os.path.join(save_dir, prefix+'.mid'))

            sample_shape = get_sample_shape(sample_size)
            save_bars(samples, size=sample_shape, file_path=save_dir, name=prefix,type_=type_)
            save_bars(samples_binary, size=sample_shape, file_path=save_dir, name=prefix+'_binary', type_=type_)
            save_bars(samples_chroma, size=sample_shape, file_path=save_dir, name=prefix+'_chroma', type_=type_)

        return samples, samples_binary, samples_chroma

    def gen_test(self, input_data, gen_dir=None, batch_size=None, num_batch=1, key='test', z=None, is_eval=True, is_save=True,type_=0):
        '''
            type_ = 'npy', 'data', 'all'
        '''
        batch_size = self.config.batch_size if not batch_size else batch_size

        gen_dir = os.path.join(self.config.exp_name, 'gen') if gen_dir is None else gen_dir

        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)

        prediction_list = []

        for bidx in range(num_batch):

            feed_dict = input_data.gen_feed_dict(idx=bidx, data_size=batch_size, key=key, z=z)

            _, sb, _ = self.run_sampler(feed_dict=feed_dict,
                                        sample_size=batch_size,
                                        save_info=is_save,
                                        save_dir=gen_dir,
                                        type_=type_)
            prediction_list.append(sb)

        result = np.concatenate(prediction_list, axis=0)
        eval_result = self.metrics.eval(result, output_type=1) if is_eval else None

        if is_save:
            np.save(os.path.join(gen_dir, 'gen.npy'), result)

        return result, eval_result