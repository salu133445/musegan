from __future__ import print_function
import numpy as np
import pretty_midi

def get_instrument(piano_roll, program_num, is_drum, velocity=100, tempo=120.0, beat_resolution=24):
    """Given a piano-roll and """
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
            note = pretty_midi.Note(velocity=velocity, pitch=note_num, start=start_times[idx], end=end_times[idx])
            # add the note to the Instrument object
            instrument.notes.append(note)
    # sort the notes by their start time
    instrument.notes.sort(key=lambda note: note.start)
    return instrument

# def write_piano_roll_to_midi(piano_roll, filename, program_num=0, is_drum=False, velocity=100, tempo=120.0,
#                              beat_resolution=24):
#     # create a PrettyMIDI object
#     midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
#     # create an Instrument object
#     instrument = pretty_midi.Instrument(program=program_num, is_drum=is_drum)
#     # set the piano roll to the Instrument object
#     set_piano_roll_to_instrument(piano_roll, instrument, velocity, tempo, beat_resolution)
#     # add the instrument to the PrettyMIDI object
#     midi.instruments.append(instrument)
#     # write MIDI file
#     midi.write(filename)

def get_midi(piano_rolls, program_nums=None, is_drum=None, velocity=100, tempo=120.0, beat_resolution=24):
    """
    XD
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
