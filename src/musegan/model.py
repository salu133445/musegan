"""This file defines the model."""
import os.path
import logging
import imageio
import numpy as np
import tensorflow as tf
from musegan.io_utils import pianoroll_to_image, vector_to_image
from musegan.io_utils import image_grid, save_pianoroll
from musegan.losses import get_adv_losses
from musegan.utils import load_component, make_sure_path_exists
LOGGER = logging.getLogger(__name__)

def get_scheduled_variable(start_value, end_value, start_step, end_step):
    """Return a scheduled decayed/growing variable."""
    if start_step > end_step:
        raise ValueError("`start_step` must be smaller than `end_step`.")
    if start_step == end_step:
        return tf.constant(start_value)
    global_step = tf.train.get_or_create_global_step()
    zero_step = tf.constant(0, dtype=global_step.dtype)
    schedule_step = tf.maximum(zero_step, global_step - start_step)
    return tf.train.polynomial_decay(
        start_value, schedule_step, end_step - start_step, end_value)

class Model:
    """Class that defines the model."""
    def __init__(self, params, name='Model'):
        self.name = name

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:

            # Save the variable scope object
            self.scope = scope

            # Build the model graph
            LOGGER.info("Building model.")
            if params.get('is_accompaniment'):
                self.gen = load_component(
                    'generator', params['nets']['generator'], 'Generator')(
                        n_tracks=params['data_shape'][-1] - 1,
                        condition_track_idx=params['condition_track_idx'])
            else:
                self.gen = load_component(
                    'generator', params['nets']['generator'], 'Generator')(
                        n_tracks=params['data_shape'][-1])
            self.dis = load_component(
                'discriminator', params['nets']['discriminator'],
                'Discriminator')(
                    n_tracks=params['data_shape'][-1],
                    beat_resolution=params['beat_resolution'])

            # Save components to a list for showing statistics
            self.components = [self.gen, self.dis]

    def __call__(self, x=None, z=None, y=None, c=None, mode=None, params=None,
                 config=None):
        if mode == 'train':
            if x is None:
                raise TypeError("`x` must not be None for 'train' mode.")
            return self.get_train_nodes(x, z, y, c, params, config)
        elif mode == 'predict':
            if z is None:
                raise TypeError("`z` must not be None for 'predict' mode.")
            return self.get_predict_nodes(z, y, c, params, config)
        raise ValueError("Unrecognized mode received. Expect 'train' or "
                         "'predict' but get {}".format(mode))

    def get_train_nodes(self, x, z=None, y=None, c=None, params=None,
                        config=None):
        """Return a dictionary of graph nodes for training."""
        LOGGER.info("Building training nodes.")
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:

            nodes = {}

            # Get or create global step
            global_step = tf.train.get_or_create_global_step()
            nodes['gen_step'] = tf.get_variable(
                'gen_step', [], tf.int32, tf.constant_initializer(0),
                trainable=False)

            # Set default latent distribution if not given
            if z is None:
                nodes['z'] = tf.truncated_normal((
                    config['batch_size'], params['latent_dim']))
            else:
                nodes['z'] = z

            # Get slope tensor (for straight-through estimators)
            nodes['slope'] = tf.get_variable(
                'slope', [], tf.float32, tf.constant_initializer(1.0),
                trainable=False)

            # --- Generator output ---------------------------------------------
            if params['use_binary_neurons']:
                if params.get('is_accompaniment'):
                    nodes['fake_x'], nodes['fake_x_preactivated'] = self.gen(
                        nodes['z'], y, c, True, nodes['slope'])
                else:
                    nodes['fake_x'], nodes['fake_x_preactivated'] = self.gen(
                        nodes['z'], y, True, nodes['slope'])
            else:
                if params.get('is_accompaniment'):
                    nodes['fake_x'] = self.gen(nodes['z'], y, c, True)
                else:
                    nodes['fake_x'] = self.gen(nodes['z'], y, True)

            # --- Slope annealing ----------------------------------------------
            if config['use_slope_annealing']:
                slope_schedule = config['slope_schedule']
                scheduled_slope = get_scheduled_variable(
                    1.0, slope_schedule['end_value'], slope_schedule['start'],
                    slope_schedule['end'])
                tf.add_to_collection(
                    tf.GraphKeys.UPDATE_OPS,
                    tf.assign(nodes['slope'], scheduled_slope))

            # --- Discriminator output -----------------------------------------
            nodes['dis_real'] = self.dis(x, y, True)
            nodes['dis_fake'] = self.dis(nodes['fake_x'], y, True)

            # ============================= Losses =============================
            LOGGER.info("Building losses.")
            # --- Adversarial losses -------------------------------------------
            nodes['gen_loss'], nodes['dis_loss'] = get_adv_losses(
                nodes['dis_real'], nodes['dis_fake'], config['gan_loss_type'])

            # --- Gradient penalties -------------------------------------------
            if config['use_gradient_penalties']:
                eps_x = tf.random_uniform(
                    [config['batch_size']] + [1] * len(params['data_shape']))
                inter_x = eps_x * x + (1.0 - eps_x) * nodes['fake_x']
                dis_x_inter_out = self.dis(inter_x, y, True)
                gradient_x = tf.gradients(dis_x_inter_out, inter_x)[0]
                slopes_x = tf.sqrt(1e-8 + tf.reduce_sum(
                    tf.square(gradient_x),
                    np.arange(1, gradient_x.get_shape().ndims)))
                gradient_penalty_x = tf.reduce_mean(tf.square(slopes_x - 1.0))
                nodes['dis_loss'] += 10.0 * gradient_penalty_x

            # Compute total loss (for logging and detecting NAN values only)
            nodes['loss'] = nodes['gen_loss'] + nodes['dis_loss']

            # ========================== Training ops ==========================
            LOGGER.info("Building training ops.")
            # --- Learning rate decay ------------------------------------------
            nodes['learning_rate'] = tf.get_variable(
                'learning_rate', [], tf.float32,
                tf.constant_initializer(config['initial_learning_rate']),
                trainable=False)
            if config['use_learning_rate_decay']:
                scheduled_learning_rate = get_scheduled_variable(
                    config['initial_learning_rate'],
                    config['learning_rate_schedule']['end_value'],
                    config['learning_rate_schedule']['start'],
                    config['learning_rate_schedule']['end'])
                tf.add_to_collection(
                    tf.GraphKeys.UPDATE_OPS,
                    tf.assign(nodes['learning_rate'], scheduled_learning_rate))

            # --- Optimizers ---------------------------------------------------
            gen_opt = tf.train.AdamOptimizer(
                nodes['learning_rate'], config['adam']['beta1'],
                config['adam']['beta2'])
            dis_opt = tf.train.AdamOptimizer(
                nodes['learning_rate'], config['adam']['beta1'],
                config['adam']['beta2'])

            # --- Training ops -------------------------------------------------
            nodes['train_ops'] = {}
            # Training op for the discriminator
            nodes['train_ops']['dis'] = dis_opt.minimize(
                nodes['dis_loss'], global_step,
                tf.trainable_variables(scope.name + '/' + self.dis.name))

            # Training ops for the generator
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            gen_step_increment = tf.assign_add(nodes['gen_step'], 1)
            with tf.control_dependencies(update_ops + [gen_step_increment]):
                nodes['train_ops']['gen'] = gen_opt.minimize(
                    nodes['gen_loss'], global_step,
                    tf.trainable_variables(scope.name + '/' + self.gen.name))

            # =========================== Summaries ============================
            LOGGER.info("Building summaries.")
            if config['save_summaries_steps'] > 0:
                with tf.name_scope('losses'):
                    tf.summary.scalar('gen_loss', nodes['gen_loss'])
                    tf.summary.scalar('dis_loss', nodes['dis_loss'])
                if config['use_learning_rate_decay']:
                    with tf.name_scope('learning_rate_decay'):
                        tf.summary.scalar(
                            'learning_rate', nodes['learning_rate'])
                if config['use_slope_annealing']:
                    with tf.name_scope('slope_annealing'):
                        tf.summary.scalar('slope', nodes['slope'])

        return nodes

    def get_predict_nodes(self, z=None, y=None, c=None, params=None,
                          config=None):
        """Return a dictionary of graph nodes for training."""
        LOGGER.info("Building prediction nodes.")
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            nodes = {'z': z}

            # Get slope tensor (for straight-through estimators)
            nodes['slope'] = tf.get_variable(
                'slope', [], tf.float32, tf.constant_initializer(1.0),
                trainable=False)

            # --- Generator output ---------------------------------------------
            if params['use_binary_neurons']:
                if params.get('is_accompaniment'):
                    nodes['fake_x'], nodes['fake_x_preactivated'] = self.gen(
                        nodes['z'], y, c, False, nodes['slope'])
                else:
                    nodes['fake_x'], nodes['fake_x_preactivated'] = self.gen(
                        nodes['z'], y, False, nodes['slope'])
            else:
                if params.get('is_accompaniment'):
                    nodes['fake_x'] = self.gen(nodes['z'], y, c, False)
                else:
                    nodes['fake_x'] = self.gen(nodes['z'], y, False)

            # ============================ Save ops ============================
            def _get_filepath(folder_name, name, suffix, ext):
                """Return the filename."""
                if suffix:
                    return os.path.join(
                        config['result_dir'], folder_name, name,
                        '{}_{}.{}'.format(name, str(suffix, 'utf8'), ext))
                return os.path.join(
                    config['result_dir'], folder_name, name,
                    '{}.{}'.format(name, ext))

            def _array_to_image(array, colormap=None):
                """Convert an array to an image array and return it."""
                if array.ndim == 2:
                    return vector_to_image(array)
                return pianoroll_to_image(array, colormap)

            # --- Save array ops -----------------------------------------------
            if config['collect_save_arrays_op']:
                def _save_array(array, suffix, name):
                    """Save the input array."""
                    filepath = _get_filepath('arrays', name, suffix, 'npy')
                    np.save(filepath, array.astype(np.float16))
                    return np.array([0], np.int32)

                arrays = {'fake_x': nodes['fake_x']}
                if params['use_binary_neurons']:
                    arrays['fake_x_preactivated'] = nodes['fake_x_preactivated']

                save_array_ops = []
                for key, value in arrays.items():
                    save_array_ops.append(tf.py_func(
                        lambda array, suffix, k=key: _save_array(
                            array, suffix, k),
                        [value, config['suffix']], tf.int32))
                    make_sure_path_exists(
                        os.path.join(config['result_dir'], 'arrays', key))
                nodes['save_arrays_op'] = tf.group(save_array_ops)

            # --- Save image ops -----------------------------------------------
            if config['collect_save_images_op']:
                def _save_image_grid(array, suffix, name):
                    image = image_grid(array, config['image_grid'])
                    filepath = _get_filepath('images', name, suffix, 'png')
                    imageio.imwrite(filepath, image)
                    return np.array([0], np.int32)

                def _save_images(array, suffix, name):
                    """Save the input image."""
                    if 'hard_thresholding' in name:
                        array = (array > 0).astype(np.float32)
                    elif 'bernoulli_sampling' in name:
                        rand_num = np.random.uniform(size=array.shape)
                        array = (.5 * (array + 1.) > rand_num)
                        array = array.astype(np.float32)
                    images = _array_to_image(array)
                    return _save_image_grid(images, suffix, name)

                def _save_colored_images(array, suffix, name):
                    """Save the input image."""
                    if 'hard_thresholding' in name:
                        array = (array > 0).astype(np.float32)
                    elif 'bernoulli_sampling' in name:
                        rand_num = np.random.uniform(size=array.shape)
                        array = (.5 * (array + 1.) > rand_num)
                        array = array.astype(np.float32)
                    images = _array_to_image(array, config['colormap'])
                    return _save_image_grid(images, suffix, name)

                images = {'fake_x': .5 * (nodes['fake_x'] + 1.)}
                if params['use_binary_neurons']:
                    images['fake_x_preactivated'] = .5 * (
                        nodes['fake_x_preactivated'] + 1.)
                else:
                    images['fake_x_hard_thresholding'] = nodes['fake_x']
                    images['fake_x_bernoulli_sampling'] = nodes['fake_x']

                save_image_ops = []
                for key, value in images.items():
                    save_image_ops.append(tf.py_func(
                        lambda array, suffix, k=key: _save_images(
                            array, suffix, k),
                        [value, config['suffix']], tf.int32))
                    save_image_ops.append(tf.py_func(
                        lambda array, suffix, k=key: _save_colored_images(
                            array, suffix, k + '_colored'),
                        [value, config['suffix']], tf.int32))
                    make_sure_path_exists(os.path.join(
                        config['result_dir'], 'images', key))
                    make_sure_path_exists(os.path.join(
                        config['result_dir'], 'images', key  + '_colored'))
                nodes['save_images_op'] = tf.group(save_image_ops)

            # --- Save pianoroll ops -------------------------------------------
            if config['collect_save_pianorolls_op']:
                def _save_pianoroll(array, suffix, name):
                    filepath = _get_filepath('pianorolls', name, suffix, 'npz')
                    if 'hard_thresholding' in name:
                        array = (array > 0)
                    elif 'bernoulli_sampling' in name:
                        rand_num = np.random.uniform(size=array.shape)
                        array = (.5 * (array + 1.) > rand_num)
                    save_pianoroll(
                        filepath, array, config['midi']['programs'],
                        list(map(bool, config['midi']['is_drums'])),
                        config['midi']['tempo'], params['beat_resolution'],
                        config['midi']['lowest_pitch'])
                    return np.array([0], np.int32)

                if params['use_binary_neurons']:
                    pianorolls = {'fake_x': nodes['fake_x'] > 0}
                else:
                    pianorolls = {
                        'fake_x_hard_thresholding': nodes['fake_x'],
                        'fake_x_bernoulli_sampling': nodes['fake_x']}

                save_pianoroll_ops = []
                for key, value in pianorolls.items():
                    save_pianoroll_ops.append(tf.py_func(
                        lambda array, suffix, k=key:
                        _save_pianoroll(array, suffix, k),
                        [value, config['suffix']], tf.int32))
                    make_sure_path_exists(
                        os.path.join(config['result_dir'], 'pianorolls', key))
                nodes['save_pianorolls_op'] = tf.group(save_pianoroll_ops)

        return nodes
