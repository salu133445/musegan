"""This script trains a model."""
import os
import logging
import argparse
from pprint import pformat
import numpy as np
import scipy.stats
import tensorflow as tf
from musegan.config import LOGLEVEL, LOG_FORMAT
from musegan.data import load_data, get_dataset, get_samples
from musegan.metrics import get_save_metric_ops
from musegan.model import Model
from musegan.utils import make_sure_path_exists, load_yaml
from musegan.utils import backup_src, update_not_none, setup_loggers
LOGGER = logging.getLogger("musegan.train")

def parse_arguments():
    """Parse and return the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', help="Directory to save all the results.")
    parser.add_argument('--params', help="Path to the model parameter file.")
    parser.add_argument('--config', help="Path to the configuration file.")
    parser.add_argument('--gpu', '--gpu_device_num', type=str, default="0",
                        help="The GPU device number to use.")
    parser.add_argument('--n_jobs', type=int,
                        help="Number of parallel calls to use for input "
                             "pipeline. Set to 1 to disable multiprocessing.")
    args = parser.parse_args()
    return args

def setup_dirs(config):
    """Setup an experiment directory structure and update the `params`
    dictionary with the directory paths."""
    # Get experiment directory structure
    config['exp_dir'] = os.path.realpath(config['exp_dir'])
    config['src_dir'] = os.path.join(config['exp_dir'], 'src')
    config['eval_dir'] = os.path.join(config['exp_dir'], 'eval')
    config['model_dir'] = os.path.join(config['exp_dir'], 'model')
    config['sample_dir'] = os.path.join(config['exp_dir'], 'samples')
    config['log_dir'] = os.path.join(config['exp_dir'], 'logs', 'train')

    # Make sure directories exist
    for key in ('log_dir', 'model_dir', 'sample_dir', 'src_dir'):
        make_sure_path_exists(config[key])

def setup():
    """Parse command line arguments, load model parameters, load configurations,
    setup environment and setup loggers."""
    # Parse the command line arguments
    args = parse_arguments()

    # Load parameters
    params = load_yaml(args.params)
    if params.get('is_accompaniment') and params.get('condition_track_idx') is None:
        raise TypeError("`condition_track_idx` cannot be None type in "
                        "accompaniment mode.")

    # Load configurations
    config = load_yaml(args.config)
    update_not_none(config, vars(args))

    # Set unspecified schedule steps to default values
    for target in (config['learning_rate_schedule'], config['slope_schedule']):
        if target['start'] is None:
            target['start'] = 0
        if target['end'] is None:
            target['end'] = config['steps']

    # Setup experiment directories and update them to configuration dictionary
    setup_dirs(config)

    # Setup loggers
    del logging.getLogger('tensorflow').handlers[0]
    setup_loggers(config['log_dir'])

    # Setup GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']

    # Backup source code
    backup_src(config['src_dir'])

    return params, config

def load_training_data(params, config):
    """Load and return the training data."""
    # Load data
    if params['is_conditional']:
        raise ValueError("Not supported yet.")
    else:
        labels = None
    LOGGER.info("Loading training data.")
    data = load_data(config['data_source'], config['data_filename'])
    LOGGER.info("Training data size: %d", len(data))

    # Build dataset
    LOGGER.info("Building dataset.")
    dataset = get_dataset(
        data, labels, config['batch_size'], params['data_shape'],
        config['use_random_transpose'], config['n_jobs'])

    # Create iterator
    if params['is_conditional']:
        train_x, train_y = dataset.make_one_shot_iterator().get_next()
    else:
        train_x, train_y = dataset.make_one_shot_iterator().get_next(), None

    return train_x, train_y

def load_or_create_samples(params, config):
    """Load or create the samples used as the sampler inputs."""
    # Load sample_z
    LOGGER.info("Loading sample_z.")
    sample_z_path = os.path.join(config['model_dir'], 'sample_z.npy')
    if os.path.exists(sample_z_path):
        sample_z = np.load(sample_z_path)
        if sample_z.shape[1] != params['latent_dim']:
            LOGGER.info("Loaded sample_z has wrong shape")
            resample = True
        else:
            resample = False
    else:
        LOGGER.info("File for sample_z not found")
        resample = True

    # Draw new sample_z
    if resample:
        LOGGER.info("Drawing new sample_z.")
        sample_z = scipy.stats.truncnorm.rvs(
            -2, 2, size=(np.prod(config['sample_grid']), params['latent_dim']))
        make_sure_path_exists(config['model_dir'])
        np.save(sample_z_path, sample_z)

    if params.get('is_accompaniment'):
        # Load sample_x
        LOGGER.info("Loading sample_x.")
        sample_x_path = os.path.join(config['model_dir'], 'sample_x.npy')
        if os.path.exists(sample_x_path):
            sample_x = np.load(sample_x_path)
            if sample_x.shape[1:] != params['data_shape']:
                LOGGER.info("Loaded sample_x has wrong shape")
                resample = True
            else:
                resample = False
        else:
            LOGGER.info("File for sample_x not found")
            resample = True

        # Draw new sample_x
        if resample:
            LOGGER.info("Drawing new sample_x.")
            data = load_data(config['data_source'], config['data_filename'])
            sample_x = get_samples(
                np.prod(config['sample_grid']), data,
                use_random_transpose = config['use_random_transpose'])
            make_sure_path_exists(config['model_dir'])
            np.save(sample_x_path, sample_x)
    else:
        sample_x = None

    return sample_x, None, sample_z

def main():
    """Main function."""
    # Setup
    logging.basicConfig(level=LOGLEVEL, format=LOG_FORMAT)
    params, config = setup()
    LOGGER.info("Using parameters:\n%s", pformat(params))
    LOGGER.info("Using configurations:\n%s", pformat(config))

    # ================================== Data ==================================
    # Load training data
    train_x, _ = load_training_data(params, config)

    # ================================= Model ==================================
    # Build model
    model = Model(params)
    if params.get('is_accompaniment'):
        train_c = tf.expand_dims(
            train_x[..., params['condition_track_idx']], -1)
        train_nodes = model(
            x=train_x, c=train_c, mode='train', params=params, config=config)
    else:
        train_nodes = model(
            x=train_x, mode='train', params=params, config=config)

    # Log number of parameters in the model
    def get_n_params(var_list):
        """Return the number of variables in a variable list."""
        return int(np.sum([np.product(
            [x.value for x in var.get_shape()]) for var in var_list]))

    LOGGER.info("Number of trainable parameters in {}: {:,}".format(
        model.name, get_n_params(tf.trainable_variables(model.name))))
    for component in model.components:
        LOGGER.info("Number of trainable parameters in {}: {:,}".format(
            component.name, get_n_params(tf.trainable_variables(
                model.name + '/' + component.name))))

    # ================================ Sampler =================================
    if config['save_samples_steps'] > 0:
        # Get sampler inputs
        sample_x, sample_y, sample_z = load_or_create_samples(params, config)

        # Create sampler configurations
        sampler_config = {
            'result_dir': config['sample_dir'],
            'suffix': tf.as_string(train_nodes['gen_step']),
            'image_grid': config['sample_grid'],
            'colormap': np.array(config['colormap']).T,
            'midi': config['midi'],
            'collect_save_arrays_op': config['save_array_samples'],
            'collect_save_images_op': config['save_image_samples'],
            'collect_save_pianorolls_op': config['save_pianoroll_samples']}

        # Get prediction nodes
        placeholder_z = tf.placeholder(tf.float32, shape=sample_z.shape)
        placeholder_y = None
        if params.get('is_accompaniment'):
            c_shape = np.append(sample_x.shape[:-1], 1)
            placeholder_c = tf.placeholder(tf.float32, shape=c_shape)
            predict_nodes = model(
                z=placeholder_z, y=placeholder_y, c=placeholder_c,
                mode='predict', params=params, config=sampler_config)
        else:
            predict_nodes = model(
                z=placeholder_z, y=placeholder_y, mode='predict', params=params,
                config=sampler_config)

        # Get sampler op
        sampler_op = tf.group([
            predict_nodes[key] for key in (
                'save_arrays_op', 'save_images_op', 'save_pianorolls_op')
            if key in predict_nodes])
        sampler_op_no_pianoroll = tf.group([
            predict_nodes[key] for key in ('save_arrays_op', 'save_images_op')
            if key in predict_nodes])

    # ================================ Metrics =================================
    if config['evaluate_steps'] > 0:
        binarized = tf.round(.5 * (predict_nodes['fake_x'] + 1.))
        save_metric_ops = get_save_metric_ops(
            binarized, params['beat_resolution'], train_nodes['gen_step'],
            config['eval_dir'])
        save_metrics_op = tf.group(save_metric_ops)

    # ========================== Training Preparation ==========================
    # Get tensorflow session config
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # Training hooks
    global_step = tf.train.get_global_step()
    steps_per_iter = config['n_dis_updates_per_gen_update'] + 1
    hooks = [tf.train.NanTensorHook(train_nodes['loss'])]

    # Tensor logger
    tensor_logger = {
        'step': train_nodes['gen_step'],
        'gen_loss': train_nodes['gen_loss'],
        'dis_loss': train_nodes['dis_loss']}
    step_logger = open(os.path.join(config['log_dir'], 'step.log'), 'w')

    # ======================= Monitored Training Session =======================
    LOGGER.info("Training start.")
    with tf.train.MonitoredTrainingSession(
        save_checkpoint_steps=config['save_checkpoint_steps'] * steps_per_iter,
        save_summaries_steps=config['save_summaries_steps'] * steps_per_iter,
        checkpoint_dir=config['model_dir'], log_step_count_steps=0,
        hooks=hooks, config=tf_config) as sess:

        # Get global step value
        step = tf.train.global_step(sess, global_step)
        if step == 0:
            step_logger.write('# step, gen_loss, dis_loss\n')

        # ============================== Training ==============================
        if step >= config['steps']:
            LOGGER.info("Global step has already exceeded total steps.")
            step_logger.close()
            return

        # Training iteration
        while step < config['steps']:

            # Train the discriminator
            if step < 10:
                n_dis_updates = 10 * config['n_dis_updates_per_gen_update']
            else:
                n_dis_updates = config['n_dis_updates_per_gen_update']
            for _ in range(n_dis_updates):
                sess.run(train_nodes['train_ops']['dis'])

            # Train the generator
            log_loss_steps = config['log_loss_steps'] or 100
            if (step + 1) % log_loss_steps == 0:
                step, _, tensor_logger_values = sess.run([
                    train_nodes['gen_step'], train_nodes['train_ops']['gen'],
                    tensor_logger])
                # Logger
                if config['log_loss_steps'] > 0:
                    LOGGER.info("step={}, {}".format(
                        tensor_logger_values['step'], ', '.join([
                            '{}={: 8.4E}'.format(key, value)
                            for key, value in tensor_logger_values.items()
                            if key != 'step'])))
                step_logger.write("{}, {: 10.6E}, {: 10.6E}\n".format(
                    tensor_logger_values['step'],
                    tensor_logger_values['gen_loss'],
                    tensor_logger_values['dis_loss']))
            else:
                step, _ = sess.run([
                    train_nodes['gen_step'], train_nodes['train_ops']['gen']])

            # Run sampler
            if ((config['save_samples_steps'] > 0)
                    and (step % config['save_samples_steps'] == 0)):
                LOGGER.info("Running sampler")
                feed_dict_sampler = {placeholder_z: sample_z}
                if params.get('is_accompaniment'):
                    feed_dict_sampler[placeholder_c] = np.expand_dims(
                        sample_x[..., params['condition_track_idx']], -1)
                if step < 3000:
                    sess.run(
                        sampler_op_no_pianoroll, feed_dict=feed_dict_sampler)
                else:
                    sess.run(sampler_op, feed_dict=feed_dict_sampler)

            # Run evaluation
            if ((config['evaluate_steps'] > 0)
                    and (step % config['evaluate_steps'] == 0)):
                LOGGER.info("Running evaluation")
                feed_dict_evaluation = {
                    placeholder_z: scipy.stats.truncnorm.rvs(-2, 2, size=(
                        np.prod(config['sample_grid']), params['latent_dim']))}
                if params.get('is_accompaniment'):
                    feed_dict_evaluation[placeholder_c] = np.expand_dims(
                        sample_x[..., params['condition_track_idx']], -1)
                sess.run(save_metrics_op, feed_dict=feed_dict_evaluation)

            # Stop training if stopping criterion suggests
            if sess.should_stop():
                break

    LOGGER.info("Training end")
    step_logger.close()

if __name__ == "__main__":
    main()
