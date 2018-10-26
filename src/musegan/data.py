"""This file contains functions for loading and preprocessing pianoroll data.
"""
import os
import logging
import numpy as np
import tensorflow as tf
from musegan.config import SHUFFLE_BUFFER_SIZE, PREFETCH_SIZE
LOGGER = logging.getLogger(__name__)

# --- Data loader --------------------------------------------------------------
def load_data_from_npy(filename):
    """Load and return the training data from a npy file."""
    data = np.load(filename)
    return data * 2. - 1.

def load_data_from_npz(filename):
    """Load and return the training data from a npz file (sparse format)."""
    with np.load(filename) as f:
        data = np.empty(f['shape'], np.float32)
        data.fill(-1.)
        data[[x for x in f['nonzero']]] = 1.
    return data

def load_data(data_source, data_filename):
    """Load and return the training data."""
    if data_source == 'sa':
        import SharedArray as sa
        return sa.attach(data_filename)
    if data_source == 'npy':
        return load_data_from_npy(data_filename)
    if data_source == 'npz':
        return load_data_from_npz(data_filename)
    raise ValueError("Expect `data_source` to be one of 'sa', 'npy', 'npz'. "
                     "But get " + str(data_source))

# --- Filename loader ----------------------------------------------------------
def findall_endswith(suffix, root):
    """Traverse `root` recursively and yield all files ending with `suffix`."""
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(suffix):
                yield os.path.join(dirpath, filename)

def load_filenames_from_dir(data_root, data_filename=None):
    """Return lits of file paths to the training and testing data. `data_root`
    will be ignored when `data_filename` is given."""
    return [os.path.realpath(filename)
            for filename in findall_endswith('.npz', data_root)]

def load_filenames(data_source, data_filename=None, data_root=None):
    """Load and return a filename list to the training data."""
    if data_source == 'filenames':
        with open(data_filename) as f:
            return [line.rstrip() for line in f]
    if data_source == 'dir':
        return load_filenames_from_dir(data_root)
    raise ValueError("Expect `data_source` to be one of 'filenames', 'dir'. "
                     "But get " + str(data_source))

# --- Dataset Utilities -------------------------------------------------------
def random_transpose(pianoroll):
    """Randomly transpose a pianoroll with [-5, 6] semitones."""
    semitone = np.random.randint(-5, 6)
    if semitone > 0:
        pianoroll[:, semitone:, 1:] = pianoroll[:, :-semitone, 1:]
        pianoroll[:, :semitone, 1:] = 0
    elif semitone < 0:
        pianoroll[:, :semitone, 1:] = pianoroll[:, -semitone:, 1:]
        pianoroll[:, semitone:, 1:] = 0
    return pianoroll

def read_pianoroll(filename, use_random_transpose=False):
    """Python code for loading a pianoroll and return it."""
    try:
        filename = str(filename, encoding='utf-8')
    except TypeError:
        pass

    # Load the pianoroll
    with np.load(filename) as f:
        pianoroll = np.zeros(f['shape'], np.float32)
        pianoroll[[x for x in f['nonzero']]] = 1.
        pianoroll = pianoroll * 2. - 1.

    # Randomly transpose the loaded pianoroll with [-5, 6] semitones
    if use_random_transpose:
        pianoroll = random_transpose(pianoroll)

    return pianoroll

def set_pianoroll_shape(pianoroll, data_shape):
    """Set the pianoroll shape and return the pianoroll."""
    pianoroll.set_shape(data_shape)
    return pianoroll

def set_label_shape(label):
    """Set the label shape and return the label."""
    label.set_shape([1])
    return label

# --- Tensorflow Dataset -------------------------------------------------------
def _gen_data(data, labels=None):
    """Data Generator."""
    if labels is None:
        for item in data:
            if np.issubdtype(data.dtype, np.bool_,):
                yield item * 2. - 1.
            else:
                yield item
    else:
        for i, item in enumerate(data):
            if np.issubdtype(data.dtype, np.bool_):
                yield (item * 2. - 1., labels[i])
            else:
                yield (item, labels[i])

def _gen_filename(filenames, labels=None):
    """Filename Generator."""
    if labels is None:
        for filename in filenames:
            yield filename
    else:
        for i, filename in enumerate(filenames):
            yield (filename, labels[i])

def get_dataset(data, labels=None, batch_size=None, data_shape=None,
                use_random_transpose=False, num_threads=1):
    """Create  and return a tensorflow dataset from an array."""
    if labels is None:
        dataset = tf.data.Dataset.from_generator(
            lambda: _gen_data(data), tf.float32)
        if use_random_transpose:
            dataset = dataset.map(
                lambda pianoroll: tf.py_func(
                    random_transpose, [pianoroll], tf.float32),
                num_parallel_calls=num_threads)
        dataset = dataset.map(lambda pianoroll: set_pianoroll_shape(
            pianoroll, data_shape), num_parallel_calls=num_threads)
    else:
        assert len(data) == len(labels), (
            "Lengths of `data` and `lables` do not match.")
        dataset = tf.data.Dataset.from_generator(
            lambda: _gen_data(data, labels), [tf.float32, tf.int32])
        if use_random_transpose:
            dataset = dataset.map(
                lambda pianoroll, label: (
                    tf.py_func(random_transpose, [pianoroll], tf.float32),
                    label),
                num_parallel_calls=num_threads)
        dataset = dataset.map(
            lambda pianoroll, label: (set_pianoroll_shape(
                pianoroll, data_shape), set_label_shape(label)),
            num_parallel_calls=num_threads)

    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).repeat().batch(batch_size)
    return dataset.prefetch(PREFETCH_SIZE)

def get_dataset_from_filenames(filenames, labels=None, batch_size=None,
                               data_shape=None, use_random_transpose=False,
                               num_threads=1):
    """Create  and return a tensorflow dataset from a filename list."""
    if labels is None:
        dataset = tf.data.Dataset.from_generator(
            lambda: _gen_filename(filenames), tf.float32)
        dataset = dataset.map(
            lambda filename: tf.py_func(read_pianoroll, [
                filename, use_random_transpose], tf.float32),
            num_parallel_calls=num_threads)
        dataset = dataset.map(lambda pianoroll: set_pianoroll_shape(
            pianoroll, data_shape), num_parallel_calls=num_threads)
    else:
        assert len(filenames) == len(labels), (
            "Lengths of `filenames` and `lables` do not match.")
        dataset = tf.data.Dataset.from_generator(
            lambda: _gen_filename(filenames, labels), [tf.float32, tf.int32])
        dataset = dataset.map(
            lambda filename, label: (tf.py_func(read_pianoroll, [
                filename, use_random_transpose], tf.float32), label),
            num_parallel_calls=num_threads)
        dataset = dataset.map(
            lambda pianoroll, label: (set_pianoroll_shape(
                pianoroll, data_shape), set_label_shape(label)),
            num_parallel_calls=num_threads)

    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).repeat().batch(batch_size)
    return dataset.prefetch(PREFETCH_SIZE)
