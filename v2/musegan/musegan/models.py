"""Class that defines the GAN model.
"""
import os.path
import time
import numpy as np
import tensorflow as tf
from musegan.model import Model
from musegan.musegan.components import Discriminator, Generator
from musegan.utils.metrics import Metrics

class GAN(Model):
    """Class that defines the first-stage (without refiner) model."""
    def __init__(self, sess, config, name='GAN', reuse=None):
        super().__init__(sess, config, name)

        print('[*] Building GAN...')
        with tf.variable_scope(name, reuse=reuse) as scope:
            self.scope = scope
            self.build()

    def build(self):
        """Build the model."""
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Create placeholders
        self.z = {}
        if self.config['net_g']['z_dim_shared'] > 0:
            self.z['shared'] = tf.placeholder(
                tf.float32, (self.config['batch_size'],
                             self.config['net_g']['z_dim_shared']), 'z_shared'
            )
        if self.config['net_g']['z_dim_private'] > 0:
            self.z['private'] = tf.placeholder(
                tf.float32, (self.config['batch_size'],
                             self.config['net_g']['z_dim_private'],
                             self.config['num_track']), 'z_private'
            )
        if self.config['net_g']['z_dim_temporal_shared'] > 0:
            self.z['temporal_shared'] = tf.placeholder(
                tf.float32, (self.config['batch_size'],
                             self.config['net_g']['z_dim_temporal_shared']),
                'z_temporal_shared'
            )
        if self.config['net_g']['z_dim_temporal_private'] > 0:
            self.z['temporal_private'] = tf.placeholder(
                tf.float32, (self.config['batch_size'],
                             self.config['net_g']['z_dim_temporal_private'],
                             self.config['num_track']), 'z_temporal_private'
            )

        data_shape = (self.config['batch_size'], self.config['num_bar'],
                      self.config['num_timestep'], self.config['num_pitch'],
                      self.config['num_track'])
        self.x = tf.placeholder(tf.bool, data_shape, 'x')
        self.x_ = tf.cast(self.x, tf.float32, 'x_')

        # Components
        self.G = Generator(self.z, self.config, name='G')
        self.test_round = self.G.tensor_out > 0.5
        self.test_bernoulli = self.G.tensor_out > tf.random_uniform(data_shape)

        self.D_fake = Discriminator(self.G.tensor_out, self.config, name='D')
        self.D_real = Discriminator(self.x_, self.config, name='D', reuse=True)
        self.components = (self.G, self.D_fake)

        # Losses
        self.g_loss, self.d_loss = self.get_adversarial_loss(Discriminator)

        # Optimizers
        with tf.variable_scope('Optimizer'):
            self.g_optimizer = self.get_optimizer()
            self.g_step = self.g_optimizer.minimize(
                self.g_loss, self.global_step, self.G.vars)

            self.d_optimizer = self.get_optimizer()
            self.d_step = self.d_optimizer.minimize(
                self.d_loss, self.global_step, self.D_fake.vars)

            # Apply weight clipping
            if self.config['gan']['type'] == 'wgan':
                with tf.control_dependencies([self.d_step]):
                    self.d_step = tf.group(
                        *(tf.assign(var, tf.clip_by_value(
                            var, -self.config['gan']['clip_value'],
                            self.config['gan']['clip_value']))
                          for var in self.D_fake.vars))

        # Metrics
        self.metrics = Metrics(self.config)

        # Saver
        self.saver = tf.train.Saver()

        # Print and save model information
        self.print_statistics()
        self.save_statistics()
        self.print_summary()
        self.save_summary()

    def train(self, x_train, train_config):
        """Train the model."""
        # Initialize sampler
        self.x_sample = x_train[np.random.choice(
            len(x_train), self.config['batch_size'], False)]
        feed_dict_sample = {self.x: self.x_sample}

        self.z_sample = {}
        for key in self.z:
            self.z_sample[key] = np.random.normal(size=self.z[key].get_shape())
            feed_dict_sample[self.z[key]] = self.z_sample[key]

        # Save samples
        self.save_samples('x_train', x_train, save_midi=True)
        self.save_samples('x_sample', self.x_sample, save_midi=True)

        # Open log files and write headers
        log_step = open(os.path.join(self.config['log_dir'], 'step.log'), 'w')
        log_batch = open(os.path.join(self.config['log_dir'], 'batch.log'), 'w')
        log_epoch = open(os.path.join(self.config['log_dir'], 'epoch.log'), 'w')
        log_step.write('# epoch, step, negative_critic_loss\n')
        log_batch.write('# epoch, batch, time, negative_critic_loss, g_loss\n')
        log_epoch.write('# epoch, time, negative_critic_loss, g_loss\n')

        # Initialize counter
        counter = 0
        num_batch = len(x_train) // self.config['batch_size']

        # Start epoch iteration
        print('{:=^80}'.format(' Training Start '))
        for epoch in range(train_config['num_epoch']):

            print('{:-^80}'.format(' Epoch {} Start '.format(epoch)))
            epoch_start_time = time.time()

            # Prepare batched training data
            z_random_batch = {}
            for key in self.z:
                z_random_batch[key] = np.random.normal(
                    size=([num_batch] + self.z[key].get_shape().as_list()))
            x_random_batch = np.random.choice(
                len(x_train), (num_batch, self.config['batch_size']), False)

            # Start batch iteration
            for batch in range(num_batch):

                feed_dict_batch = {self.x: x_train[x_random_batch[batch]]}
                for key in self.z:
                    feed_dict_batch[self.z[key]] = z_random_batch[key][batch]

                if (counter < 25) or (counter % 500 == 0):
                    num_critics = 100
                else:
                    num_critics = 5

                batch_start_time = time.time()

                # Update networks
                for _ in range(num_critics):
                    _, d_loss = self.sess.run([self.d_step, self.d_loss],
                                              feed_dict_batch)
                    log_step.write("{}, {:14.6f}\n".format(
                        self.get_global_step_str(), -d_loss
                    ))

                _, d_loss, g_loss = self.sess.run(
                    [self.g_step, self.d_loss, self.g_loss], feed_dict_batch
                )
                log_step.write("{}, {:14.6f}\n".format(
                    self.get_global_step_str(), -d_loss
                ))

                time_batch = time.time() - batch_start_time

                # Print iteration summary
                if train_config['verbose']:
                    if batch < 1:
                        print("epoch |   batch   |  time  |    - D_loss    |"
                              "     G_loss")
                    print("  {:2d}  | {:4d}/{:4d} | {:6.2f} | {:14.6f} | "
                          "{:14.6f}".format(epoch, batch, num_batch, time_batch,
                                            -d_loss, g_loss))

                log_batch.write("{:d}, {:d}, {:f}, {:f}, {:f}\n".format(
                    epoch, batch, time_batch, -d_loss, g_loss
                ))

                # run sampler
                if train_config['sample_along_training']:
                    if counter%100 == 0 or (counter < 300 and counter%20 == 0):
                        self.run_sampler(self.G.tensor_out, feed_dict_sample,
                                         False)
                        self.run_sampler(self.test_round, feed_dict_sample,
                                         (counter > 500), postfix='test_round')
                        self.run_sampler(self.test_bernoulli, feed_dict_sample,
                                         (counter > 500),
                                         postfix='test_bernoulli')

                # run evaluation
                if train_config['evaluate_along_training']:
                    if counter%10 == 0:
                        self.run_eval(self.test_round, feed_dict_sample,
                                      postfix='test_round')
                        self.run_eval(self.test_bernoulli, feed_dict_sample,
                                      postfix='test_bernoulli')

                counter += 1

            # print epoch info
            time_epoch = time.time() - epoch_start_time

            if not train_config['verbose']:
                if epoch < 1:
                    print("epoch |   time   |    - D_loss    |     G_loss")
                print("  {:2d}  | {:8.2f} | {:14.6f} | {:14.6f}".format(
                    epoch, time_epoch, -d_loss, g_loss))

            log_epoch.write("{:d}, {:f}, {:f}, {:f}\n".format(
                epoch, time_epoch, -d_loss, g_loss
            ))

            # save checkpoints
            self.save()

        print('{:=^80}'.format(' Training End '))
        log_step.close()
        log_batch.close()
        log_epoch.close()
