# Herman Dong, 2017.06.20 v1
# Last modified 2017.07.13
"""
This module
"""
from __future__ import print_function
import numpy as np
import pretty_midi

def set_piano_roll_to_instrument(piano_roll, instrument, velocity=100, tempo=120.0, beat_resolution=24):
    # calculate time per pixel
    tpp = 60.0/tempo/float(beat_resolution)
    # create piano_roll_search that captures note onsets and offsets
    piano_roll = piano_roll.reshape((piano_roll.shape[0] * piano_roll.shape[1], piano_roll.shape[2]))
    piano_roll_diff = np.concatenate((np.zeros((1, 128), dtype=int), piano_roll, np.zeros((1, 128), dtype=int)))
    piano_roll_search = np.diff(piano_roll_diff.astype(int), axis=0)
    # iterate through all possible(128) pitches
    for note_num in range(128):
        # search for note onsets and offsets
        start_idx = (piano_roll_search[:, note_num] > 0).nonzero()
        start_time = tpp*(start_idx[0].astype(float))
        end_idx = (piano_roll_search[:, note_num] < 0).nonzero()
        end_time = tpp*(end_idx[0].astype(float))
        # iterate through all the searched notes
        for idx in range(len(start_time)):
            # create an Note object with corresponding note number, start time and end time
            note = pretty_midi.Note(velocity=velocity, pitch=note_num, start=start_time[idx], end=end_time[idx])
            # add the note to the Instrument object
            instrument.notes.append(note)
    # sort the notes by their start time
    instrument.notes.sort(key=lambda note: note.start)

def write_piano_roll_to_midi(piano_roll, filename, program_num=0, is_drum=False, velocity=100, tempo=120.0,
                             beat_resolution=24):
    # create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    # create an Instrument object
    instrument = pretty_midi.Instrument(program=program_num, is_drum=is_drum)
    # set the piano roll to the Instrument object
    set_piano_roll_to_instrument(piano_roll, instrument, velocity, tempo, beat_resolution)
    # add the instrument to the PrettyMIDI object
    midi.instruments.append(instrument)
    # write MIDI file
    midi.write(filename)

def write_piano_rolls_to_midi(piano_rolls, filename, program_nums=None, is_drum=None, velocity=100, tempo=120.0,
                              beat_resolution=24):
    if len(piano_rolls) != len(program_nums) or len(piano_rolls) != len(is_drum):
        print("Error: piano_rolls and program_nums/is_drum have different sizes...")
        return False
    if program_nums is None:
        program_nums = [0] * len(piano_rolls)
    if is_drum is None:
        is_drum = [False] * len(piano_rolls)
    # create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    # iterate through all the input instruments
    for idx, piano_roll in enumerate(piano_rolls):
        # create an Instrument object
        instrument = pretty_midi.Instrument(program=program_nums[idx], is_drum=is_drum[idx])
        # set the piano roll to the Instrument object
        set_piano_roll_to_instrument(piano_roll, instrument, velocity, tempo, beat_resolution)
        # add the instrument to the PrettyMIDI object
        midi.instruments.append(instrument)
    # write MIDI file
    midi.write(filename)

def run_demo():
    # create a piano roll of two bar with two tracks (piano and drums)
    piano_rolls = [np.zeros((4, 96, 128), dtype=bool), np.zeros((4, 96, 128), dtype=bool)]
    program_nums = [0, 0]
    is_drum = [False, True]
    # create a C(G) chord in the first(second) bar for the piano track
    piano_rolls[0][0, :, 60] = 1 # note C4
    piano_rolls[0][0, :, 64] = 1 # note E4
    piano_rolls[0][0, :, 67] = 1 # note G4
    piano_rolls[0][1, :, 62] = 1 # note D4
    piano_rolls[0][1, :, 67] = 1 # note G4
    piano_rolls[0][1, :, 71] = 1 # note B4
    piano_rolls[0][2, :, 64] = 1 # note E4
    piano_rolls[0][2, :, 69] = 1 # note A4
    piano_rolls[0][2, :, 72] = 1 # note C5
    piano_rolls[0][3, :, 60] = 1 # note C4
    piano_rolls[0][3, :, 65] = 1 # note E4
    piano_rolls[0][3, :, 69] = 1 # note A4
    # create a 8-beat pattern for the drum track
    for beat_idx in range(8):
        piano_rolls[1][:, beat_idx*12, 42] = 1 # closed hit-hat
    for beat_idx in range(4):
        piano_rolls[1][:, beat_idx*24, 36] = 1 # bass drum
        piano_rolls[1][:, (beat_idx+2)*24, 40] = 1 # snare drum
    piano_rolls[1][:, 60, 36] = 1 # bass drum
    # write the piano roll into a midi file
    write_piano_rolls_to_midi(piano_rolls, filename='test.midi', program_nums=program_nums, is_drum=is_drum)

if __name__ == '__main__':
    run_demo()
