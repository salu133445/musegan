"""This file defines the metrics used to evaluate the generated results."""
import os
import numpy as np
import tensorflow as tf
from musegan.utils import make_sure_path_exists

# --- Utilities ----------------------------------------------------------------
def to_chroma(pianoroll):
    """Return the chroma features (not normalized)."""
    if pianoroll.get_shape().ndims != 5:
        raise ValueError("Input tensor must have 5 dimensions.")
    remainder = pianoroll.get_shape()[3] % 12
    if remainder:
        pianoroll = tf.pad(
            pianoroll, ((0, 0), (0, 0), (0, 0), (0, 12 - remainder), (0, 0)))
    reshaped = tf.reshape(
        pianoroll, (-1, pianoroll.get_shape()[1], pianoroll.get_shape()[2], 12,
                    pianoroll.get_shape()[3] // 12 + int(remainder > 0),
                    pianoroll.get_shape()[4]))
    return tf.reduce_sum(reshaped, 4)

# --- Metrics ------------------------------------------------------------------
def empty_bar_rate(tensor):
    """Return the ratio of empty bars to the total number of bars."""
    if tensor.get_shape().ndims != 5:
        raise ValueError("Input tensor must have 5 dimensions.")
    return tf.reduce_mean(
        tf.cast(tf.reduce_any(tensor > 0.5, (2, 3)), tf.float32), (0, 1))

def n_pitches_used(tensor):
    """Return the number of unique pitches used per bar."""
    if tensor.get_shape().ndims != 5:
        raise ValueError("Input tensor must have 5 dimensions.")
    return tf.reduce_mean(tf.reduce_sum(tf.count_nonzero(tensor, 3), 2), [0, 1])

def qualified_note_rate(tensor, threshold=2):
    """Return the ratio of the number of the qualified notes (notes longer than
    `threshold` (in time step)) to the total number of notes in a piano-roll."""
    if tensor.get_shape().ndims != 5:
        raise ValueError("Input tensor must have 5 dimensions.")
    def _qualified_note_rate(array, threshold):
        """Return the ratio of the number of the qualified notes (notes longer
        than `threshold` (in time step)) to the total number of notes in a
        piano-roll."""
        n_tracks = array.shape[-1]
        reshaped = array.reshape(-1, array.shape[1] * array.shape[2],
                                 array.shape[3], array.shape[4])
        padded = np.pad(reshaped.astype(int), ((0, 0), (1, 1), (0, 0), (0, 0)),
                        'constant')
        diff = np.diff(padded, axis=1)
        transposed = diff.transpose(3, 0, 1, 2).reshape(n_tracks, -1)
        onsets = (transposed > 0).nonzero()
        offsets = (transposed < 0).nonzero()
        n_qualified_notes = np.array([np.count_nonzero(
            offsets[1][(offsets[0] == i)] - onsets[1][(onsets[0] == i)]
            >= threshold) for i in range(n_tracks)], np.float32)
        n_onsets = np.array([np.count_nonzero(onsets[1][(onsets[0] == i)])
                             for i in range(n_tracks)], np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            return n_qualified_notes / n_onsets
    return tf.py_func(lambda array: _qualified_note_rate(array, threshold),
                      [tensor], tf.float32)

def polyphonic_rate(tensor, threshold=2):
    """Return the ratio of the number of time steps where the number of pitches
    being played is larger than `threshold` to the total number of time steps"""
    if tensor.get_shape().ndims != 5:
        raise ValueError("Input tensor must have 5 dimensions.")
    n_poly = tf.count_nonzero((tf.count_nonzero(tensor, 3) > threshold), 2)
    return tf.reduce_mean((n_poly / tensor.get_shape()[2]), [0, 1])

def drum_in_pattern_rate(tensor):
    """Return the drum_in_pattern_rate metric value."""
    if tensor.get_shape().ndims != 4:
        raise ValueError("Input tensor must have 4 dimensions.")

    def _drum_pattern_mask(n_timesteps, tolerance=0.1):
        """Return a drum pattern mask with the given tolerance."""
        if n_timesteps not in (96, 48, 24, 72, 36, 64, 32, 16):
            raise ValueError("Unsupported number of timesteps for the drum in "
                             "pattern metric.")
        if n_timesteps == 96:
            drum_pattern_mask = np.tile(
                [1., tolerance, 0., 0., 0., tolerance], 16)
        elif n_timesteps == 48:
            drum_pattern_mask = np.tile([1., tolerance, tolerance], 16)
        elif n_timesteps == 24:
            drum_pattern_mask = np.tile([1., tolerance, tolerance], 8)
        elif n_timesteps == 72:
            drum_pattern_mask = np.tile(
                [1., tolerance, 0., 0., 0., tolerance], 12)
        elif n_timesteps == 36:
            drum_pattern_mask = np.tile([1., tolerance, tolerance], 12)
        elif n_timesteps == 64:
            drum_pattern_mask = np.tile([1., tolerance, 0., tolerance], 16)
        elif n_timesteps == 32:
            drum_pattern_mask = np.tile([1., tolerance], 16)
        elif n_timesteps == 16:
            drum_pattern_mask = np.tile([1., tolerance], 8)
        return drum_pattern_mask

    drum_pattern_mask = _drum_pattern_mask(tensor.get_shape()[2])
    drum_pattern_mask = tf.constant(
        drum_pattern_mask.reshape(1, 1, tensor.get_shape()[2]), tf.float32)
    n_in_pattern = tf.reduce_sum(drum_pattern_mask * tf.reduce_sum(tensor, 3))
    n_notes = tf.count_nonzero(tensor, dtype=tf.float32)
    return tf.cond((n_notes > 0), lambda: (n_in_pattern / n_notes), lambda: 0.)

def in_scale_rate(tensor):
    """Return the in_scale_rate metric value."""
    if tensor.get_shape().ndims != 5:
        raise ValueError("Input tensor must have 5 dimensions.")
    if tensor.get_shape()[3] != 12:
        raise ValueError("Input tensor must be a chroma tensor.")

    def _scale_mask(key=3):
        """Return a scale mask for the given key. Default to C major scale."""
        a_scale_mask = np.array([[[1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]]], bool)
        return np.expand_dims(np.roll(a_scale_mask, -key, 2), -1)

    scale_mask = tf.constant(_scale_mask(), tf.float32)
    in_scale = tf.reduce_sum(scale_mask * tf.reduce_sum(tensor, 2), [0, 1, 2])
    return in_scale / tf.reduce_sum(tensor, (0, 1, 2, 3))

def harmonicity(tensor, beat_resolution):
    """Return the harmonicity metric value"""
    if tensor.get_shape().ndims != 5:
        raise ValueError("Input tensor must have 5 dimensions.")
    if tensor.get_shape()[3] != 12:
        raise ValueError("Input tensor must be a chroma tensor.")

    def _tonal_matrix(r1=1.0, r2=1.0, r3=0.5):
        """Compute and return a tonal matrix for computing the tonal distance
        [1]. Default argument values are set as suggested by the paper.

        [1] Christopher Harte, Mark Sandler, and Martin Gasser. Detecting
        harmonic change in musical audio. In Proc. ACM MM Workshop on Audio and
        Music Computing Multimedia, 2006.
        """
        tonal_matrix = np.empty((6, 12))
        tonal_matrix[0] = r1 * np.sin(np.arange(12) * (7. / 6.) * np.pi)
        tonal_matrix[1] = r1 * np.cos(np.arange(12) * (7. / 6.) * np.pi)
        tonal_matrix[2] = r2 * np.sin(np.arange(12) * (3. / 2.) * np.pi)
        tonal_matrix[3] = r2 * np.cos(np.arange(12) * (3. / 2.) * np.pi)
        tonal_matrix[4] = r3 * np.sin(np.arange(12) * (2. / 3.) * np.pi)
        tonal_matrix[5] = r3 * np.cos(np.arange(12) * (2. / 3.) * np.pi)
        return tonal_matrix

    def _to_tonal_space(tensor):
        """Return the tensor in tonal space where chroma features are normalized
        per beat."""
        tonal_matrix = tf.constant(_tonal_matrix(), tf.float32)
        beat_chroma = tf.reduce_sum(tf.reshape(
            tensor, (-1, beat_resolution, 12, tensor.get_shape()[4])), 1)
        beat_chroma = beat_chroma / tf.reduce_sum(beat_chroma, 1, True)
        reshaped = tf.reshape(tf.transpose(beat_chroma, (1, 0, 2)), (12, -1))
        return tf.reshape(
            tf.matmul(tonal_matrix, reshaped), (6, -1, tensor.get_shape()[4]))

    mapped = _to_tonal_space(tensor)
    expanded1 = tf.expand_dims(mapped, -1)
    expanded2 = tf.expand_dims(mapped, -2)
    tonal_dist = tf.norm(expanded1 - expanded2, axis=0)
    return tf.reduce_mean(tonal_dist, 0)

# --- Metric op utilities ------------------------------------------------------
def get_metric_ops(tensor, beat_resolution):
    """Return a dictionary of metric ops."""
    if tensor.get_shape().ndims != 5:
        raise ValueError("Input tensor must have 5 dimensions.")
    chroma = to_chroma(tensor)
    metric_ops = {
        'empty_bar_rate': empty_bar_rate(tensor),
        'n_pitches_used': n_pitches_used(tensor),
        'n_pitch_classes_used': n_pitches_used(chroma),
        'polyphonic_rate': polyphonic_rate(tensor),
        'in_scale_ratio': in_scale_rate(chroma),
        'qualified_note_rate': qualified_note_rate(tensor),
        'drum_in_pattern_rate': drum_in_pattern_rate(tensor[..., 0]),
        'harmonicity': harmonicity(chroma[..., 1:], beat_resolution),
    }
    return metric_ops

def get_save_metric_ops(tensor, beat_resolution, step, result_dir, suffix=None):
    """Return save metric ops."""
    # Get metric ops
    metric_ops = get_metric_ops(tensor, beat_resolution)

    # Make sure paths exist
    for key in metric_ops:
        make_sure_path_exists(os.path.join(result_dir, key))

    def _save_array(array, step, name):
        """Save the input array."""
        suffix_ = step if suffix is None else suffix
        filepath = os.path.join(
            result_dir, name, '{}_{}.npy'.format(name, suffix_))
        np.save(filepath, array)
        return np.array([0], np.int32)

    save_metric_ops = {}
    for key, value in metric_ops.items():
        save_metric_ops[key] = tf.py_func(
            lambda array, step, k=key: _save_array(array, step, k),
            [value, step], tf.int32)

    return save_metric_ops
