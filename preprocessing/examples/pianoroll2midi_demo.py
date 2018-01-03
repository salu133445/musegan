import numpy as np
import pretty_midi
import pianoroll2midi

if __name__ == "__main__":
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
    midi = get_midi(piano_rolls, program_nums=program_nums, is_drum=is_drum)
    midi.write('test.midi')
