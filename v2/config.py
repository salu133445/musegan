"""Define configuration variables in experiment, model and training levels.

Quick Setup
===========
Change the values in the dictionary `SETUP` for a quick setup.
Documentation is provided right after each key.

Configuration
=============
More configuration options are providedin as a dictionary `CONFIG`.
`CONFIG['exp']`, `CONFIG['data']`, `CONFIG['model']`, `CONFIG['train']` and
`CONFIG['tensorflow']` define experiment-, data-, model-, training-,
TensorFlow-related configuration variables, respectively.

Note that the automatically-determined experiment name is based only on the
values defined in the dictionary `SETUP`, so remember to provide the experiment
name manually if you have changed the configuration so that you won't overwrite
existing experiment directories.
"""
import os
import shutil
import distutils.dir_util
import importlib
import numpy as np
import tensorflow as tf

# Quick setup
SETUP = {
    'model': 'musegan',
    # {'musegan', 'bmusegan'}
    # The model to use. Currently support MuseGAN and BinaryMuseGAN models.

    'exp_name': None,
    # The experiment name. A folder with the same name will be created in 'exp/'
    # directory where all the experiment-related files will be saved. None to
    # determine automatically. Note that the automatically-determined experiment
    # name is based only on the values defined in the dictionary `SETUP`, so
    # remember to provide the experiment name manually when you modify any other
    # configuration variable (so that you won't overwrite a trained model).

    'prefix': 'lastfm_alternative',
    # Prefix for the experiment name. Useful when training with different
    # training data to avoid replacing the previous experiment outputs.

    'training_data': 'lastfm_alternative_8b_phrase.npy',
    # Path to the training data. The training data can be loaded from a npy
    # file in the hard disk or from the shared memory using SharedArray package.
    # Note that the data will be reshaped to (-1, num_bar, num_timestep,
    # num_pitch, num_track) and remember to set these variable to proper values,
    # which are defined in `CONFIG['model']`.

    'training_data_location': 'hd',
    # Location of the training data. 'hd' to load from a npy file stored in the
    # hard disk. 'sa' to load from shared memory using SharedArray package.

    'gpu': '0',
    # The GPU index in os.environ['CUDA_VISIBLE_DEVICES'] to use.

    'preset_g': 'hybrid',
    # MuseGAN: {'composer', 'jamming', 'hybrid'}
    # BinaryMuseGAN: {'proposed', 'proposed_small'}
    # Use a preset network architecture for the generator or set to None and
    # setup `CONFIG['model']['net_g']` to define the network architecture.

    'preset_d': 'proposed',
    # {'proposed', 'proposed_small', 'ablated', 'baseline', None}
    # Use a preset network architecture for the discriminator or set to None
    # and setup `CONFIG['model']['net_d']` to define the network architecture.

    'pretrained_dir': None,
    # The directory containing the pretrained model. None to retrain the
    # model from scratch.

    'verbose': True,
    # True to print each batch details to stdout. False to print once an epoch.

    'sample_along_training': True,
    # True to generate samples along the training process. False for nothing.

    'evaluate_along_training': True,
    # True to run evaluation along the training process. False for nothing.

    # ------------------------- For BinaryMuseGAN only -------------------------
    'two_stage_training': True,
    # True to train the model in a two-stage training setting. False to
    # train the model in an end-to-end manner.

    'training_phase': 'first_stage',
    # {'first_stage', 'second_stage'}
    # The training phase in a two-stage training setting. Only effective
    # when `two_stage_training` is True.

    'first_stage_dir': None,
    # The directory containing the pretrained first-stage model. None to
    # determine automatically (assuming using default `exp_name`). Only
    # effective when two_stage_training` is True and `training_phase` is
    # 'second_stage'.

    'joint_training': False,
    # True to train the generator and the refiner jointly. Only effective
    # when `two_stage_training` is True and `training_phase` is 'second_stage'.

    'preset_r': 'proposed_bernoulli',
    # {'proposed_round', 'proposed_bernoulli'}
    # Use a preset network architecture for the refiner or set to None and
    # setup `CONFIG['model']['net_r']` to define the network architecture.
}

CONFIG = {}

#===============================================================================
#=========================== TensorFlow Configuration ==========================
#===============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = SETUP['gpu']
CONFIG['tensorflow'] = tf.ConfigProto()
CONFIG['tensorflow'].gpu_options.allow_growth = True

#===============================================================================
#========================== Experiment Configuration ===========================
#===============================================================================
CONFIG['exp'] = {
    'model': None,
    'exp_name': None,
    'pretrained_dir': None,
    'two_stage_training': None, # For BinaryMuseGAN only
    'first_stage_dir': None, # For BinaryMuseGAN only
}

for key in ('model', 'pretrained_dir'):
    if CONFIG['exp'][key] is None:
        CONFIG['exp'][key] = SETUP[key]

if SETUP['model'] == 'musegan':
    # Set default experiment name
    if CONFIG['exp']['exp_name'] is None:
        if SETUP['exp_name'] is not None:
            CONFIG['exp']['exp_name'] = SETUP['exp_name']
        else:
            CONFIG['exp']['exp_name'] = '_'.join(
                (SETUP['prefix'], 'g', SETUP['preset_g'], 'd',
                 SETUP['preset_d']))

if SETUP['model'] == 'bmusegan':
    if CONFIG['exp']['two_stage_training'] is None:
        CONFIG['exp']['two_stage_training'] = SETUP['two_stage_training']
    # Set default experiment name
    if CONFIG['exp']['exp_name'] is None:
        if SETUP['exp_name'] is not None:
            CONFIG['exp']['exp_name'] = SETUP['exp_name']
        elif not SETUP['two_stage_training']:
            CONFIG['exp']['exp_name'] = '_'.join(
                (SETUP['prefix'], 'end2end', 'g', SETUP['preset_g'], 'd',
                 SETUP['preset_d'], 'r', SETUP['preset_r']))
        elif SETUP['training_phase'] == 'first_stage':
            CONFIG['exp']['exp_name'] = '_'.join(
                (SETUP['prefix'], SETUP['training_phase'], 'g',
                 SETUP['preset_g'], 'd', SETUP['preset_d']))
        elif SETUP['training_phase'] == 'second_stage':
            if SETUP['joint_training']:
                CONFIG['exp']['exp_name'] = '_'.join(
                    (SETUP['prefix'], SETUP['training_phase'], 'joint', 'g',
                     SETUP['preset_g'], 'd', SETUP['preset_d'], 'r',
                     SETUP['preset_r']))
            else:
                CONFIG['exp']['exp_name'] = '_'.join(
                    (SETUP['prefix'], SETUP['training_phase'], 'g',
                     SETUP['preset_g'], 'd', SETUP['preset_d'], 'r',
                     SETUP['preset_r']))
    # Set default first stage model directory
    if CONFIG['exp']['first_stage_dir'] is None:
        if SETUP['first_stage_dir'] is not None:
            CONFIG['exp']['first_stage_dir'] = SETUP['first_stage_dir']
        else:
            CONFIG['exp']['first_stage_dir'] = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'exp',
                '_'.join((SETUP['prefix'], 'first_stage', 'g',
                          SETUP['preset_g'], 'd', SETUP['preset_d'])),
                'checkpoints')

#===============================================================================
#============================= Data Configuration ==============================
#===============================================================================
CONFIG['data'] = {
    'training_data': None,
    'training_data_location': None,
}

for key in ('training_data', 'training_data_location'):
    if CONFIG['data'][key] is None:
        CONFIG['data'][key] = SETUP[key]

#===============================================================================
#=========================== Training Configuration ============================
#===============================================================================
CONFIG['train'] = {
    'num_epoch': 10,
    'verbose': None,
    'sample_along_training': None,
    'evaluate_along_training': None,
    'two_stage_training': None, # For BinaryMuseGAN only
    'training_phase': None, # For BinaryMuseGAN only
    'slope_annealing_rate': 1.1, # For BinaryMuseGAN only
}

for key in ('verbose', 'sample_along_training', 'evaluate_along_training'):
    if CONFIG['train'][key] is None:
        CONFIG['train'][key] = SETUP[key]

if SETUP['model'] == 'bmusegan' and CONFIG['train']['training_phase'] is None:
    CONFIG['train']['training_phase'] = SETUP['training_phase']

#===============================================================================
#============================= Model Configuration =============================
#===============================================================================
CONFIG['model'] = {
    'joint_training': None, # For BinaryMuseGAN only

    # Parameters
    'batch_size': 32, # Note: tf.layers.conv3d_transpose requires a fixed batch
                      # size in TensorFlow < 1.6
    'gan': {
        'type': 'wgan-gp', # 'gan', 'wgan', 'wgan-gp'
        'clip_value': .01,
        'gp_coefficient': 10.
    },
    'optimizer': {
        # Parameters for the Adam optimizers
        'lr': .002,
        'beta1': .5,
        'beta2': .9,
        'epsilon': 1e-8
    },

    # Data
    'num_bar': 4,
    'num_beat': 4,
    'num_pitch': 84,
    'num_track': 8,
    'num_timestep': 96,
    'beat_resolution': 24,
    'lowest_pitch': 24, # MIDI note number of the lowest pitch in data tensors

    # Tracks
    'track_names': (
        'Drums', 'Piano', 'Guitar', 'Bass', 'Ensemble', 'Reed', 'Synth Lead',
        'Synth Pad'
    ),
    'programs': (0, 0, 24, 32, 48, 64, 80, 88),
    'is_drums': (True, False, False, False, False, False, False, False),

    # Network architectures (define them here if not using the presets)
    'net_g': None,
    'net_d': None,
    'net_r': None, # For BinaryMuseGAN only

    # Playback
    'pause_between_samples': 96,
    'tempo': 90.,

    # Samples
    'num_sample': 16,
    'sample_grid': (2, 8),

    # Metrics
    'metric_map': np.array([
        # indices of tracks for the metrics to compute
        [True] * 8, # empty bar rate
        [True] * 8, # number of pitch used
        [False] + [True] * 7, # qualified note rate
        [False] + [True] * 7, # polyphonicity
        [False] + [True] * 7, # in scale rate
        [True] + [False] * 7, # in drum pattern rate
        [False] + [True] * 7  # number of chroma used
    ], dtype=bool),
    'tonal_distance_pairs': [(1, 2)], # pairs to compute the tonal distance
    'scale_mask': list(map(bool, [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])),
    'drum_filter': np.tile([1., .1, 0., 0., 0., .1], 16),
    'tonal_matrix_coefficient': (1., 1., .5),

    # Directories
    'checkpoint_dir': None,
    'sample_dir': None,
    'eval_dir': None,
    'log_dir': None,
    'src_dir': None,
}

if SETUP['model'] == 'bmusegan' and CONFIG['model']['joint_training'] is None:
    CONFIG['model']['joint_training'] = SETUP['joint_training']

# Import preset network architectures
if CONFIG['model']['net_g'] is None:
    IMPORTED = importlib.import_module(
        '.'.join(('musegan', SETUP['model'], 'presets', 'generator',
                  SETUP['preset_g'])))
    CONFIG['model']['net_g'] = IMPORTED.NET_G

if CONFIG['model']['net_d'] is None:
    IMPORTED = importlib.import_module(
        '.'.join(('musegan', SETUP['model'], 'presets', 'discriminator',
                  SETUP['preset_d'])))
    CONFIG['model']['net_d'] = IMPORTED.NET_D

if SETUP['model'] == 'bmusegan' and CONFIG['model']['net_r'] is None:
    IMPORTED = importlib.import_module(
        '.'.join(('musegan.bmusegan.presets', 'refiner', SETUP['preset_r'])))
    CONFIG['model']['net_r'] = IMPORTED.NET_R

# Set default directories
for kv_pair in (('checkpoint_dir', 'checkpoints'), ('sample_dir', 'samples'),
                ('eval_dir', 'eval'), ('log_dir', 'logs'), ('src_dir', 'src')):
    if CONFIG['model'][kv_pair[0]] is None:
        CONFIG['model'][kv_pair[0]] = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'exp', SETUP['model'],
            CONFIG['exp']['exp_name'], kv_pair[1])

#===============================================================================
#=================== Make directories & Backup source code =====================
#===============================================================================
# Make sure directories exist
for path in (CONFIG['model']['checkpoint_dir'], CONFIG['model']['sample_dir'],
             CONFIG['model']['eval_dir'], CONFIG['model']['log_dir'],
             CONFIG['model']['src_dir']):
    if not os.path.exists(path):
        os.makedirs(path)

# Backup source code
for path in os.listdir(os.path.dirname(os.path.realpath(__file__))):
    if os.path.isfile(path):
        if path.endswith('.py'):
            shutil.copyfile(os.path.basename(path),
                            os.path.join(CONFIG['model']['src_dir'],
                                         os.path.basename(path)))

distutils.dir_util.copy_tree(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'musegan'),
    os.path.join(CONFIG['model']['src_dir'], 'musegan')
)
