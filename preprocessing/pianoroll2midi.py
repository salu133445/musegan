"""Utility functions for conversion from pianoroll to midi"""

from __future__ import print_function
import numpy as np
import pretty_midi

def get_instrument(piano_roll, program_num=0, is_drum=False, velocity=100, tempo=120.0, beat_resolution=24):
    """
    Convert a piano-roll to a pretty_midi.Instrument object.

    Parameters
    ----------
    piano_roll : np.ndarray of int (or bool)
        The size is (num_time_step, num_pitches). If dtype is int, the values
        represent the velocity. If dtype is bool, the parameter 'velocity' is
        used for all notes.
    program_num : int
        MIDI program number, in [0, 127].
    is_drum : bool
        Indicates whether the piano-roll is a drum track.
    velocity : int
        Velocity of notes, in [0, 127].
    tempo: float
        Tempo in beat per minute (bpm).
    beat_resolution: int
        The resolution of a beat used in the input piano-roll.

    Returns
    -------
    instrument : pretty_midi.Instrument
        The converted result, a pretty_midi.Instrument object.
    """
    # calculate time per pixel
    tpp = 60.0/tempo/float(beat_resolution)
    # create piano_roll_search that captures note onsets and offsets
    piano_roll = piano_roll.reshape((piano_roll.shape[0] * piano_roll.shape[1], piano_roll.shape[2]))
    piano_roll_diff = np.concatenate((np.zeros((1, 128), dtype=int), piano_roll, np.zeros((1, 128), dtype=int)))
    piano_roll_search = np.diff(piano_roll_diff.astype(int), axis=0)
    # create an Instrument object
    instrument = pretty_midi.Instrument(program=program_num, is_drum=is_drum)
    # iterate through all possible(128) pitches
    for note_num in range(128):
        # search for note onsets and offsets
        start_idxs = (piano_roll_search[:, note_num] > 0).nonzero()
        start_times = tpp*(start_idxs[0].astype(float))
        end_idxs = (piano_roll_search[:, note_num] < 0).nonzero()
        end_times = tpp*(end_idxs[0].astype(float))
        # iterate through all the searched notes
        for idx, start_time in enumerate(start_times):
            # create an Note object with corresponding note number, start time and end time
            note = pretty_midi.Note(velocity=velocity, pitch=note_num, start=start_time, end=end_times[idx])
            # add the note to the Instrument object
            instrument.notes.append(note)
    # sort the notes by their start time
    instrument.notes.sort(key=lambda note: note.start)
    return instrument

def get_midi(piano_rolls, program_nums=None, is_drum=None, velocity=100, tempo=120.0, beat_resolution=24):
    """
    Convert a multi-track piano-roll to a pretty_midi.PrettyMIDI object.

    Parameters
    ----------
    piano_rolls : list of np.ndarray of int (or bool)
        The size of each item is (num_time_step, num_pitches). If dtype is
        int, the values represent the velocity. If dtype is bool, the
        parameter 'velocity' is used for all notes.
    program_nums : list of int
        MIDI program number, in [0, 127], of the corresponding piano-roll.
        The length should be the same as the parameter 'piano_rolls'.
    is_drum : list of bool
        Indicates whether the corresponding piano-roll is a drum track. The
        length should be the same as the parameter 'piano_rolls'.
    velocity : int
        Velocity of notes, in [0, 127].
    tempo: float
        Tempo in beat per minute (bpm).
    beat_resolution: int
        The resolution of a beat used in the input piano-roll.

    Returns
    -------
    pm : pretty_midi.PrettyMIDI
        The converted result, a pretty_midi.PrettyMIDI object.
    """
    if len(piano_rolls) != len(program_nums):
        raise ValueError("piano_rolls and program_nums should have the same length")
    if len(piano_rolls) != len(is_drum):
        raise ValueError("piano_rolls and is_drum should have the same length")
    if program_nums is None:
        program_nums = [0] * len(piano_rolls)
    if is_drum is None:
        is_drum = [False] * len(piano_rolls)
    # create a PrettyMIDI object
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    # iterate through all the input instruments
    for idx, piano_roll in enumerate(piano_rolls):
        instrument = get_instrument(piano_roll, program_nums[idx], is_drum[idx], velocity, tempo, beat_resolution)
        pm.instruments.append(instrument)
    return pm
