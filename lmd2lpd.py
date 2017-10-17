# Herman Dong 2017.5.18 v1
# Last modified: Herman Dong 2017.09.27 v2
"""
[How to use]
    (1) set the correct DATASET_ROOT, SEARCH_ROOT, RESULTS_PATH constant
    (2) set the correct SCORE_FILE path constant
    (3) set the desired FILETYPE constant
    (4) set the correct CONFIDENCE_TH, BEAT_RESOLUTION constant
    (5) run the following
            python2 midi2pianoroll.py
[Notes]
    Only the midi file with highest confidence score is chosen.
    occurrence ratio: the ratio of the number of bars that the instrument shows up to the number of bars in total
    average notes simultaneously: the average number of notes occurs simultaneously
    max notes simultaneously: the maximum number of notes occurs simultaneously
    rhythm complexity: average number of notes(onset only) in a bar
    pitch complexity: average number of notes with different pitches
"""

from __future__ import print_function
import os
import json
import numpy as np
from config import settings
from midi2pianoroll import midi_to_pianorolls

if settings['filetype'] == 'npz':
    import scipy.sparse


def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

def get_midi_path(msd_id, midi_md5):
    """Given an MSD ID and MIDI MD5, return path to a MIDI file.
    kind should be one of 'matched' or 'aligned'. """
    return os.path.join(settings['dataset_path'], msd_id_to_dirs(msd_id), midi_md5 + '.mid')

def make_sure_path_exists(path):
    """Create all intermediate-level directories if the given path doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def get_instrument_filename(instrument_info, identifier):
    """Given a pretty_midi.Instrument class instance and the identifier, return the filename of the instrument."""
    family_name = instrument_info[str(identifier)]['family_name']
    program_name = instrument_info[str(identifier)]['program_name']
    return '_'.join([str(identifier), family_name.replace(' ', '-'), program_name.replace(' ', '-')])

def save_ndarray(ndarray, dir_path, filename):
    """Save a numpy ndarray to the given directory with filetype given in settings file."""
    filepath = os.path.join(dir_path, filename + settings['filetype'])
    if settings['filetype'] == 'npy':
        np.save(filepath, ndarray)
    elif settings['filetype'] == 'npz':
        sparse_ndarray = scipy.sparse.csc_matrix(ndarray)
        scipy.sparse.save_npz(filepath, sparse_ndarray)
    elif settings['filetype'] == 'csv':
        np.savetxt(filepath, ndarray, delimiter=',')
    else:
        raise TypeError("unknown file type, allowed values are 'npy', 'npz' and 'csv'")

def save_ndarray_dict(ndarray_dict, dir_path):
    """Save a dict of numpy ndarrays to the given directory."""
    for key, value in ndarray_dict.iteritems():
        save_ndarray(value, dir_path, filename=key)

def save_piano_rolls(piano_rolls, dir_path, instrument_info, postfix=None):
    """Save piano-rolls to files named <instrument_id>_<family_name>_<program_name>."""
    for idx, piano_roll in enumerate(piano_rolls):
        filename = get_instrument_filename(instrument_info, idx) + postfix
        save_ndarray(piano_roll, dir_path, filename)

def get_piano_roll_statistics(piano_roll, onset_array, midi_data):
    """Get the statistics of a piano-roll, onset_array and midi_data should be given."""
    # occurrence beat ratio
    sum_rhythm_bool = piano_roll.sum(dtype=bool, axis=1)
    occ_beats = sum_rhythm_bool.reshape((settings['beat_resolution'], -1)).sum(dtype=bool, axis=0)
    occurrence_beat_ratio = occ_beats / float(midi_data['num_beats'])
    # occurrence bar ratio
    sum_rhythm_bool = piano_roll.sum(dtype=bool, axis=1)
    if midi_data['time_signature'] is not None:
        num_step_bar = settings['beat_resolution'] * int(midi_data['time_signature'][0])
        occ_bars = sum_rhythm_bool.reshape((num_step_bar, -1)).sum(dtype=bool, axis=0)
        occurrence_bar_ratio = occ_bars / float(midi_data['num_bars'])
    else:
        occurrence_bar_ratio = None
    # average notes simultaneously
    sum_rhythm_int = piano_roll.sum(axis=1)
    avg_notes_simultaneously = sum_rhythm_int.sum() / float(sum_rhythm_int.sum()) if sum_rhythm_int.sum() > 0 else 0.0
    # max notes simultaneously
    max_notes_simultaneously = max(sum_rhythm_int)
    # rhythm complexity
    rhythm_complexity = float(np.sum(onset_array)) / float(occ_bars) if occ_bars > 0 else 0.0
    # pitch complexity
    if midi_data['time_signature'] is not None:
        sum_pitch_bar = piano_roll.reshape(-1, settings['beat_resolution']*midi_data['time_signature'][-1], 128) \
                                  .sum(axis=1)
        pitch_complexity_bar = (sum_pitch_bar > 0).sum(axis=1)
        pitch_complexity = np.sum(pitch_complexity_bar) / float(occ_bars) if occ_bars > 0 else 0.0
    return {'occurrence beat ratio': occurrence_beat_ratio,
            'occurrence bar ratio': occurrence_bar_ratio,
            'average notes simultaneously': avg_notes_simultaneously,
            'max notes simultaneously': max_notes_simultaneously,
            'rhythm complexity': rhythm_complexity,
            'pitch complexity': pitch_complexity}

def main():
    # create a dictionary of the selected midi files
    midi_dict = {}
    # traverse from dataset root directory
    for dirpath, _, filenames in os.walk(settings['dataset_path']):
        # collect midi files into a list
        midi_filenames = [f for f in filenames if f.endswith('.mid')]
        # skip the current folder if no midi files found
        if not midi_filenames:
            continue
        # iterate through all midi files found
        for midi_filename in midi_filenames:
            # get the msd_id and midi_md5
            midi_md5 = os.path.splitext(midi_filename)[0]
            msd_id = os.path.basename(os.path.normpath(dirpath))
            # get the midi filepath
            midi_filepath = os.path.join(dirpath, midi_filename)
            # get the path to save the results
            result_song_dir = os.path.join(settings['result_path'], msd_id_to_dirs(msd_id))
            result_midi_dir = os.path.join(result_song_dir, midi_md5)
            # make sure the result directory exists
            make_sure_path_exists(result_midi_dir)
            # convert the midi file into piano-rolls
            piano_rolls, midi_data = midi_to_pianorolls(midi_filepath)
            # save the piano-rolls into files
            save_piano_rolls(piano_rolls, result_midi_dir, midi_data['instrument_info'])
            # save the onset arrays into files in a subfolder named 'onset_arrays'
            save_piano_rolls(midi_data['onset_arrays'], os.path.join(result_midi_dir, 'onset_arrays'),
                             midi_data['instrument_info'], postfix='_onset')
            # save the midi arrays to files
            save_ndarray_dict(midi_data['midi_arrays'], result_midi_dir)
            # add a key value pair storing the midi_md5 of the selected midi file
            midi_data['midi_info']['midi_md5'] = midi_md5
            # store the midi_info_dict in the midi dictionary
            midi_dict[msd_id] = midi_data['midi_info']
            # save the instrument dictionary into a json file
            with open(os.path.join(dirpath, 'instruments.json'), 'w') as outfile:
                json.dump(midi_data['instrument_info'], outfile)
    # save the midi dict into a json file
    with open(os.path.join(settings['result_path'], 'midi_dict.json'), 'w') as outfile:
        json.dump(midi_dict, outfile)

if __name__ == "__main__":
    main()
