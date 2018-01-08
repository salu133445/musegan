from __future__ import print_function
import os
import json
import warnings
import numpy as np
from config import settings
from utils import make_sure_path_exists, get_lpd_file_dir
from utils import save_npz, save_dict_to_json
from midi2pianoroll import midi_to_pianorolls

if settings['multicore'] > 1:
    import joblib

warnings.filterwarnings('ignore')

def get_piano_roll_statistics(piano_roll, onset_array, midi_data):
    """Get the statistics of a piano-roll."""
    # get the binarized version of the piano_roll
    piano_roll_bool = (piano_roll > 0)
    # occurrence beat ratio
    sum_rhythm_bool = piano_roll_bool.sum(dtype=bool, axis=1)
    occ_beats = sum_rhythm_bool.reshape((settings['beat_resolution'], -1)).sum(dtype=bool, axis=0)
    occurrence_beat_ratio = occ_beats / float(midi_data['num_beats'])
    # occurrence bar ratio
    sum_rhythm_bool = piano_roll_bool.sum(dtype=bool, axis=1)
    if midi_data['time_signature'] is not None:
        num_step_bar = settings['beat_resolution'] * int(midi_data['time_signature'][0])
        occ_bars = sum_rhythm_bool.reshape((num_step_bar, -1)).sum(dtype=bool, axis=0)
        occurrence_bar_ratio = occ_bars / float(midi_data['num_bars'])
    else:
        occurrence_bar_ratio = None
    # average notes simultaneously
    sum_rhythm_int = piano_roll_bool.sum(axis=1)
    avg_notes_simultaneously = sum_rhythm_int.sum() / float(sum_rhythm_int.sum()) if sum_rhythm_int.sum() > 0 else 0.0
    # max notes simultaneously
    max_notes_simultaneously = max(sum_rhythm_int)
    # rhythm complexity
    rhythm_complexity = float(np.sum(onset_array)) / float(occ_bars) if occ_bars > 0 else 0.0
    # pitch complexity
    if midi_data['time_signature'] is not None:
        sum_pitch_bar = piano_roll_bool.reshape(-1, settings['beat_resolution']*midi_data['time_signature'][-1], 128) \
                                  .sum(axis=1)
        pitch_complexity_bar = (sum_pitch_bar > 0).sum(axis=1)
        pitch_complexity = np.sum(pitch_complexity_bar) / float(occ_bars) if occ_bars > 0 else 0.0
    return {'occurrence beat ratio': occurrence_beat_ratio,
            'occurrence bar ratio': occurrence_bar_ratio,
            'average notes simultaneously': avg_notes_simultaneously,
            'max notes simultaneously': max_notes_simultaneously,
            'rhythm complexity': rhythm_complexity,
            'pitch complexity': pitch_complexity}

def converter(filepath):
    """Given the midi_filepath, convert it to piano-rolls and save the
    piano-rolls along with other side products. Return a key value pair for
    storing midi info to a dictionary."""
    # convert the midi file into piano-rolls
    try:
        piano_rolls, onset_rolls, info_dict = midi_to_pianorolls(filepath, beat_resolution=settings['beat_resolution'])
    except:
        return None
    # get the msd_id and midi_md5 the path to save the results
    midi_md5 = os.path.splitext(os.path.basename(filepath))[0]
    if settings['dataset'] == 'matched':
        msd_id = os.path.basename(os.path.dirname(filepath))
        result_dir = get_lpd_file_dir(midi_md5, msd_id)
    else:
        result_dir = get_lpd_file_dir(midi_md5)
    # save the piano-rolls an the onset-rolls into files
    make_sure_path_exists(result_dir)
    save_npz(os.path.join(result_dir, 'piano_rolls.npz'), sparse_matrices=piano_rolls)
    save_npz(os.path.join(result_dir, 'onset_rolls.npz'), sparse_matrices=onset_rolls)
    # save the midi arrays into files
    sparse_matrices_keys = ['tempo_array', 'beat_array', 'downbeat_array']
    sparse_matrices = {key: value for key, value in info_dict['midi_arrays'].iteritems() if key in sparse_matrices_keys}
    arrays = {key: value for key, value in info_dict['midi_arrays'].iteritems() if key not in sparse_matrices_keys}
    save_npz(os.path.join(result_dir, 'arrays.npz'), arrays=arrays, sparse_matrices=sparse_matrices)
    # save the instrument dictionary into a json file
    save_dict_to_json(info_dict['instrument_info'], os.path.join(result_dir, 'instruments.json'))
    if settings['dataset'] == 'matched':
        return (msd_id, {midi_md5: info_dict['midi_info']})
    else:
        return (midi_md5, info_dict['midi_info'])

def main():
    # traverse from dataset root directory and search for midi files
    midi_filepaths = []
    for dirpath, _, filenames in os.walk(settings['dataset_path']):
        for filename in filenames:
            if filename.endswith('.mid'):
                midi_filepaths.append(os.path.join(dirpath, filename))
    # parallelize the converter if in multicore mode
    if settings['multicore'] > 1:
        kv_pairs = joblib.Parallel(n_jobs=settings['multicore'], verbose=5)(
            joblib.delayed(converter)(midi_filepath) for midi_filepath in midi_filepaths)
        # save the midi dict into a json file
        kv_pairs = [kv_pair for kv_pair in kv_pairs if kv_pair is not None]
        midi_dict = {}
        for key in set([kv_pair[0] for kv_pair in kv_pairs]):
            midi_dict[key] = {}
        for kv_pair in kv_pairs:
            midi_dict[kv_pair[0]].update(kv_pair[1])
    else:
        midi_dict = {}
        for midi_filepath in midi_filepaths:
            kv_pair = converter(midi_filepath)
            if kv_pair is not None:
                midi_dict[kv_pair[0]] = kv_pair[1]
    # save the midi dict into a json file
    save_dict_to_json(midi_dict, os.path.join(settings['lpd_path'], 'midis.json'))
    print('{} MIDI files have been converted to multi-track piano-rolls.'.format(len(midi_dict)))

if __name__ == "__main__":
    main()
