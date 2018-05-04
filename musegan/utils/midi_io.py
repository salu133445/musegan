"""Utilities for writing piano-rolls to MIDI files.
"""
import numpy as np
from pypianoroll import Multitrack, Track

def write_midi(filepath, pianorolls, program_nums=None, is_drums=None,
               track_names=None, velocity=100, tempo=120.0, beat_resolution=24):
    """
    Write the given piano-roll(s) to a single MIDI file.

    Arguments
    ---------
    filepath : str
        Path to save the MIDI file.
    pianorolls : np.array, ndim=3
        The piano-roll array to be written to the MIDI file. Shape is
        (num_timestep, num_pitch, num_track).
    program_nums : int or list of int
        MIDI program number(s) to be assigned to the MIDI track(s). Available
        values are 0 to 127. Must have the same length as `pianorolls`.
    is_drums : list of bool
        Drum indicator(s) to be assigned to the MIDI track(s). True for
        drums. False for other instruments. Must have the same length as
        `pianorolls`.
    track_names : list of str
        Track name(s) to be assigned to the MIDI track(s).
    """
    if not np.issubdtype(pianorolls.dtype, np.bool_):
        raise TypeError("Support only binary-valued piano-rolls")
    if isinstance(program_nums, int):
        program_nums = [program_nums]
    if isinstance(is_drums, int):
        is_drums = [is_drums]

    if pianorolls.shape[2] != len(program_nums):
        raise ValueError("`pianorolls` and `program_nums` must have the same"
                         "length")
    if pianorolls.shape[2] != len(is_drums):
        raise ValueError("`pianorolls` and `is_drums` must have the same"
                         "length")
    if program_nums is None:
        program_nums = [0] * len(pianorolls)
    if is_drums is None:
        is_drums = [False] * len(pianorolls)

    multitrack = Multitrack(beat_resolution=beat_resolution, tempo=tempo)
    for idx in range(pianorolls.shape[2]):
        if track_names is None:
            track = Track(pianorolls[..., idx], program_nums[idx],
                          is_drums[idx])
        else:
            track = Track(pianorolls[..., idx], program_nums[idx],
                          is_drums[idx], track_names[idx])
        multitrack.append_track(track)
    multitrack.write(filepath)

def save_midi(filepath, phrases, config):
    """
    Save a batch of phrases to a single MIDI file.

    Arguments
    ---------
    filepath : str
        Path to save the image grid.
    phrases : list of np.array
        Phrase arrays to be saved. All arrays must have the same shape.
    pause : int
        Length of pauses (in timestep) to be inserted between phrases.
        Default to 0.
    """
    if not np.issubdtype(phrases.dtype, np.bool_):
        raise TypeError("Support only binary-valued piano-rolls")

    reshaped = phrases.reshape(-1, phrases.shape[1] * phrases.shape[2],
                               phrases.shape[3], phrases.shape[4])
    pad_width = ((0, 0), (0, config['pause_between_samples']),
                 (config['lowest_pitch'],
                  128 - config['lowest_pitch'] - config['num_pitch']),
                 (0, 0))
    padded = np.pad(reshaped, pad_width, 'constant')
    pianorolls = padded.reshape(-1, padded.shape[2], padded.shape[3])

    write_midi(filepath, pianorolls, config['programs'], config['is_drums'],
               tempo=config['tempo'])
