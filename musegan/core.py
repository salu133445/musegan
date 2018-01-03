from __future__ import division
from __future__ import print_function
import os
import time
from glob import glob
import numpy as np
import tensorflow as tf
from six.moves import xrange
from sklearn.utils import shuffle
import SharedArray as sa
from libs.ops import *
from libs.utils import *
from eval.metrics import *
from components import *
from config import *
# from config.default_128_r_off_y_off import *

class MuseGAN(object):
    def __init__(self,sess):

        print ('{:=^120}'.format(' Building MidiNet '))
        self.sess = sess
        self.build_model()

    def build_model(self):

        # create global step variable and increment op
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.global_step_increment = tf.assign(self.global_step, self.global_step+1)

        # metrics
        print ('*build metrics...')
        drum_filter = np.tile([1,0.3,0,0,0,0.3], 16)
        scale_mask = [1., 0., 1., 0., 1., 1., 0., 1., 0., 1., 0., 1.]
        inter_pair = [(0,2), (0,3), (0,4), (2,3), (2,4), (3,4)]
        track_names = ['Bass', 'Drum', 'Guitar', 'Other', 'String']
        eval_map = np.array([
                        [1, 1, 1, 1, 1],  # metric_is_empty_bar
                        [1, 1, 1, 1, 1],  # metric_num_pitch_used
                        [1, 0, 1, 1, 1],  # metric_too_short_note_ratio
                        [1, 0, 1, 1, 1],  # metric_polyphonic_ratio
                        [1, 0, 1, 1, 1],  # metric_in_scale
                        [0, 1, 0, 0, 0],  # metric_drum_pattern
                        [1, 0, 1, 1, 1]   # metric_num_chroma_used
                    ])

        self.M = Metrics(eval_map=eval_map,
                 inter_pair=inter_pair,
                 drum_filter=drum_filter,
                 scale_mask=scale_mask,
                 track_names=track_names,
                 is_build_graph=False)

        # create generator (G)
        self.Model = model_config(NowBarConfig)

        self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        self.summary = tf.summary.merge(self.summaries)
        self.summary_d = tf.summary.merge([s for s in self.summaries if '/d/' in s.name])
        self.summary_g = tf.summary.merge([s for s in self.summaries if '/g/' in s.name])
        self.summary_image = tf.summary.merge([s for s in self.summaries if '/prediction/' in s.name])


        """ Model Information """
        print('{:=^120}'.format(' Model Information '))
        file_model_info = open(os.path.join(DIR_LOG, 'model_info.txt'), 'w')
        num_parameter_g = np.sum([np.product([x.value for x in var.get_shape()]) for var in self.Model.g_vars])
        num_parameter_d = np.sum([np.product([x.value for x in var.get_shape()]) for var in self.Model.d_vars])
        num_parameter_all = np.sum([np.product([x.value for x in var.get_shape()]) for var in tf.trainable_variables()])
        print('# of parameters in G (generator)                 |', num_parameter_g)
        print('# of parameters in D (discriminator)             |', num_parameter_d)
        print('# of parameters in total                         |', num_parameter_all)
        print('# of parameters in G (generator)                 |', num_parameter_g, file=file_model_info)
        print('# of parameters in D (discriminator)             |', num_parameter_d, file=file_model_info)
        print('# of parameters in total                         |', num_parameter_all, file=file_model_info)

        """ Saver """
        self.saver = tf.train.Saver()
        self.saver_g = tf.train.Saver(self.Model.g_vars, max_to_keep=30)
        self.saver_d = tf.train.Saver(self.Model.d_vars, max_to_keep=30)
        self.saver_dict = {'midinet': self.saver, 'G':self.saver_g, 'D':self.saver_d}
        print( '{:=^120}'.format('Done!'))

    def init_all(self):
        print('*initializing variables...')
        tf.global_variables_initializer().run()

    def train(self):

        # dataset size and type

        self.x_train = sa.attach(DATA_X_TRA_NAME)
        self.x_val = sa.attach(DATA_X_VAL_NAME)
        self.y_train = sa.attach(DATA_Y_TRA_NAME)
        self.y_val = sa.attach(DATA_Y_VAL_NAME)
        print ('train size: {} ({})| condition size: {} ({})'.format(len(self.x_train),self.x_train.dtype,
                                                                                len(self.y_train), self.y_train.dtype))
        # Mtrics on Training Data
        print ('\n\n[Metrics]')
        self.M.eval(self.x_train[:256], quiet=False)

        ################################################## Log Files ###################################################
        # open files for save logs
        file_train_epoch = open(os.path.join(DIR_LOG, 'train_epoch.txt'), 'w')
        file_train_batch = open(os.path.join(DIR_LOG, 'train_batch.txt'), 'w')
        folder_name = os.path.join(DIR_LOG)
        self.writer_train_epoch = tf.summary.FileWriter(os.path.join(folder_name, 'train_epoch'), self.sess.graph)
        self.writer_train_batch = tf.summary.FileWriter(os.path.join(folder_name, 'train_batch'), self.sess.graph)
        self.writer_val_epoch = tf.summary.FileWriter(os.path.join(folder_name, 'val_epoch'), self.sess.graph)
        if SAVE_IMAGE_SUMMARY:
            self.writer_images = tf.summary.FileWriter(os.path.join(folder_name, 'images'), self.sess.graph)

        ############################################ Sampler Initialization ############################################
        # save training data samples to image files
        print( '*saving training data samples...')
        random_picked_sample = np.random.choice(xrange(len(self.x_train)), SAMPLE_SIZE, replace=False)
        x_sample = self.x_train[random_picked_sample]*2. - 1.
        y_sample = np.ones_like(self.y_train[random_picked_sample])
        z_sample = np.random.normal(0, 0.1, [SAMPLE_SIZE, Z_DIM]).astype(np.float32)
        z_sample_intra = np.random.normal(0, 0.1, [SAMPLE_SIZE, Z_DIM, TRACK_DIM]).astype(np.float32)

        # feed_dict to edit
        # feed_dict_sample = {self.z: z_sample, self.x: x_sample, self.y: y_sample}
        feed_dict_sample = {self.Model.z_intra:z_sample_intra, self.Model.z_inter: z_sample, self.Model.x: x_sample}

        save_bars(x_sample, SAMPLE_SHAPE, file_path=os.path.join(DIR_SAMPLE, 'Train(random).png'))


        ################################################### Training ###################################################
        # initial counter and timer
        print('{:=^120}'.format(' Training Start '))
        print('# epoch, time, negative_critic_loss, g_loss', file=file_train_epoch)
        print('# epoch, batch, time, negative_critic_loss, g_loss', file=file_train_batch)
        counter = 0

        num_batch_train = min(TRAIN_SIZE, len(self.x_train) // BATCH_SIZE)
        for epoch in range(EPOCH):

            print ('{:-^120}'.format(' Epoch {} Start '.format(epoch)))
            epoch_start_time = time.time()

            for batch in range(num_batch_train): #num_batch_train

                st = BATCH_SIZE*batch
                ed = BATCH_SIZE*(batch+1)

                batch_start_time = time.time()

                z_train_batch = np.random.normal(0, 0.1, [BATCH_SIZE, Z_DIM]).astype(np.float32)
                z_train_batch_intra = np.random.normal(0, 0.1, [BATCH_SIZE, Z_DIM, TRACK_DIM]).astype(np.float32)
                y_train_batch = self.y_train[st:ed]
                x_train_batch = self.x_train[st:ed] * 2. - 1.

                # feed_dict to edit
                # feed_dict_batch = {self.z: z_train_batch, self.y: y_train_batch, self.x: x_train_batch}
                feed_dict_batch = {self.Model.z_intra: z_train_batch_intra,
                            self.Model.z_inter: z_train_batch, self.Model.x: x_train_batch}

                # num of critic
                if counter < 25 or counter % 500 == 0:
                    num_iters_D = 100
                else:
                    num_iters_D = 5


                for j in range(num_iters_D-1):
                    self.sess.run(self.Model.d_optim, feed_dict=feed_dict_batch)

                _, summary_str = self.sess.run([self.Model.d_optim, self.Model.summary_d], feed_dict=feed_dict_batch)
                self.writer_train_batch.add_summary(summary_str, global_step=self.global_step.eval())

                    # update generator network(G)
                _, summary_str = self.sess.run([self.Model.g_optim, self.Model.summary_g], feed_dict=feed_dict_batch)
                self.writer_train_batch.add_summary(summary_str, global_step=self.global_step.eval())

                # evaluation
                samples_binary = self.sess.run([self.Model.prediction_binary], feed_dict=feed_dict_batch)
                score_matrix , score_pair = self.M.eval(samples_binary[0], quiet=True)

                # compute losses
                d_loss_eval = self.Model.d_loss.eval(feed_dict_batch) * -1.
                g_loss_eval = self.Model.g_loss.eval(feed_dict_batch)

                # collect batch info
                self.M.collect(score_matrix , score_pair)
                self.M.collect_loss({'d':d_loss_eval, 'g':g_loss_eval})

                # print and save batch info
                if PRINT_BATCH:
                    print( '---{}--- epoch: {:2d} | batch: {:4d}/{:4d} | time: {:6.2f} ============================'\
                                                .format(EXP_NAME+'_GPU_'+GPU_NUM, epoch, batch, num_batch_train, time.time() - batch_start_time))

                    print (d_loss_eval)
                    print (g_loss_eval)

                print (epoch, batch, d_loss_eval, g_loss_eval, file=file_train_batch)

                if (counter%500) == 0:
                    self.M.save_history(DIR_LOG)

                    samples_binary = self.sess.run([self.Model.prediction_binary], feed_dict=feed_dict_batch)
                    self.M.eval(samples_binary[0], quiet=False)
                    self.run_bar_sampler(generator=self.Model,
                                        feed_dict=feed_dict_sample,
                                        prefix='train_{:03d}'.format(counter),
                                        save_midi=True)

                counter += 1
                self.sess.run(self.global_step_increment)

            # print epoch info
            print ('epoch: {:2d} | time: {:6.2f} |'.format(epoch, time.time() - epoch_start_time))
            print (epoch, d_loss_eval, g_loss_eval, file= file_train_epoch)

            # run sampler
            if (epoch%10) == 0:
                self.save(DIR_CHECKPOINT, component='GD', global_step=self.global_step)

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

    def run_sampler(self, generator, feed_dict=None, prefix='sample', save_midi=True):

        # run sampler
        print('*running sampler...')
        samples, samples_binary, samples_chroma = self.sess.run([generator.prediction, generator.prediction_binary,
                                                                 generator.prediction_chroma], feed_dict=feed_dict)

        # save results to image files
        save_bars(samples, size=SAMPLE_SHAPE, file_path=os.path.join(DIR_SAMPLE, prefix+'.png'))
        save_bars(samples_binary, size=SAMPLE_SHAPE, file_path=os.path.join(DIR_SAMPLE, prefix+'_binary.png'))
        save_bars(samples_chroma, size=SAMPLE_SHAPE, file_path=os.path.join(DIR_SAMPLE, prefix+'_chroma.png'))

        if save_midi:
            print('*saving midi files...')
            save_midis(samples_binary, file_path=os.path.join(DIR_SAMPLE, prefix+'.mid'))
    def run_bar_sampler(self, generator, feed_dict=None, prefix='sample', save_midi=True):

        # run sampler
        print ('*running sampler...')
        samples, samples_binary, samples_chroma = self.sess.run([generator.prediction, generator.prediction_binary,
                                                                 generator.prediction_chroma], feed_dict=feed_dict)


        # save results to image files
        save_bars(samples, size=SAMPLE_SHAPE, file_path=os.path.join(DIR_SAMPLE, prefix+'.png'))
        save_bars(samples_binary, size=SAMPLE_SHAPE, file_path=os.path.join(DIR_SAMPLE, prefix+'_binary.png'))
        save_bars(samples_chroma, size=SAMPLE_SHAPE, file_path=os.path.join(DIR_SAMPLE, prefix+'_chroma.png'))

        if save_midi:
            print('*saving midi files...')
            save_midis(samples_binary, file_path=os.path.join(DIR_SAMPLE, prefix+'.mid'))

