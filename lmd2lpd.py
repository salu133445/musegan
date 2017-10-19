from __future__ import print_function
import os
import json
import warnings
import numpy as np
import scipy.sparse
from config import settings
from midi2pianoroll import midi_to_pianorolls

if settings['multicores'] > 1:
    import joblib

warnings.filterwarnings('ignore')

def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

def get_midi_path(msd_id, midi_md5):
    """Given an MSD ID and MIDI MD5, return path to a MIDI file.
    kind should be one of 'matched' or 'aligned'. """
    return os.path.join(settings['dataset_path'], msd_id_to_dirs(msd_id), midi_md5 + '.mid')

def make_sure_path_exists(path):
    """Create all intermediate-level directories if the given path not exist"""
    while True:
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                break
            except OSError as e:
                pass

def save_npz(filepath, arrays=None, sparse_matrices=None):
    """"Save the given matrices into one single '.npz' file."""
    arrays_dict = {}
    if arrays:
        if isinstance(arrays, dict):
            arrays_dict = arrays
        else:
            # if arg arrays is given as other iterable, set to default name, 'arr_0', 'arr_1', ...
            for idx, array in enumerate(arrays):
                arrays_dict['arr_' + idx] = array
    if sparse_matrices:
        if isinstance(sparse_matrices, dict):
            # convert sparse matrices to sparse representations of arrays if any
            for sparse_matrix_name, sparse_matrix in sparse_matrices.iteritems():
                csc_matrix = scipy.sparse.csc_matrix(sparse_matrix)
                # emmbed indices into filenames for future use when loading
                arrays_dict['_'.join([sparse_matrix_name, 'csc_data'])] = csc_matrix.data
                arrays_dict['_'.join([sparse_matrix_name, 'csc_indices'])] = csc_matrix.indices
                arrays_dict['_'.join([sparse_matrix_name, 'csc_indptr'])] = csc_matrix.indptr
                arrays_dict['_'.join([sparse_matrix_name, 'csc_shape'])] = csc_matrix.shape
        else:
            # convert sparse matrices to sparse representations of arrays if any
            for idx, sparse_matrix in enumerate(sparse_matrices):
                csc_matrix = scipy.sparse.csc_matrix(sparse_matrix)
                # emmbed indices into filenames for future use when loading
                arrays_dict['_'.join([str(idx), 'csc_data'])] = csc_matrix.data
                arrays_dict['_'.join([str(idx), 'csc_indices'])] = csc_matrix.indices
                arrays_dict['_'.join([str(idx), 'csc_indptr'])] = csc_matrix.indptr
                arrays_dict['_'.join([str(idx), 'csc_shape'])] = csc_matrix.shape
    # save to a compressed npz file
    if not filepath.endswith('.npz'):
        filepath = filepath + '.npz'
    np.savez_compressed(filepath, **arrays_dict)

def load_npz(filepath):
    """Load the file and return the numpy arrays and scipy csc_matrices."""
    with np.load(filepath) as loaded:
        # serach for non-sparse arrays
        arrays_name = [filename for filename in loaded.files if "_csc_" not in filename]
        arrays = {array_name: loaded[array_name] for array_name in arrays_name}
        # serach for csc matrices
        csc_matrices_name = sorted([filename for filename in loaded.files if "_csc_" in filename])
        csc_matrices = {}
        if csc_matrices_name:
            for idx in range(len(csc_matrices_name)/4):
                csc_matrix_name = csc_matrices_name[4*idx][:-9] # remove tailing 'csc_data'
                csc_matrices[csc_matrix_name] = scipy.sparse.csc_matrix((loaded[csc_matrices_name[4*idx]],
                                                                         loaded[csc_matrices_name[4*idx+1]],
                                                                         loaded[csc_matrices_name[4*idx+2]]),
                                                                        shape=loaded[csc_matrices_name[4*idx+3]])
        return arrays, csc_matrices

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

def save_dict_to_json(data, filepath):
    """Save the data dictionary to the given filepath."""
    with open(filepath, 'w') as outfile:
        json.dump(data, outfile)

def converter(filepath):
    """Given the midi_filepath, convert it to piano-rolls and save the
    piano-rolls along with other side products. Return a key value pair for
    storing midi info to a dictionary."""
    # get the msd_id and midi_md5
    midi_md5 = os.path.splitext(os.path.basename(filepath))[0]
    if settings['link_to_msd']:
        msd_id = os.path.basename(os.path.dirname(filepath))
    # convert the midi file into piano-rolls
    try:
        piano_rolls, onset_rolls, info_dict = midi_to_pianorolls(filepath, beat_resolution=settings['beat_resolution'])
    except Exception as err:
        return None
    # get the path to save the results
    if settings['link_to_msd']:
        result_midi_dir = os.path.join(settings['result_path'], msd_id_to_dirs(msd_id), midi_md5)
    else:
        result_midi_dir = os.path.join(settings['result_path'], midi_md5[0])
    # save the piano-rolls an the onset-rolls into files
    make_sure_path_exists(result_midi_dir)
    save_npz(os.path.join(result_midi_dir, 'piano_rolls.npz'), sparse_matrices=piano_rolls)
    save_npz(os.path.join(result_midi_dir, 'onset_rolls.npz'), sparse_matrices=onset_rolls)
    # save the midi arrays into files
    sparse_matrices_keys = ['tempo_array', 'beat_array', 'downbeat_array']
    sparse_matrices = {key: value for key, value in info_dict['midi_arrays'].iteritems() if key in sparse_matrices_keys}
    arrays = {key: value for key, value in info_dict['midi_arrays'].iteritems() if key not in sparse_matrices_keys}
    save_npz(os.path.join(result_midi_dir, 'arrays.npz'), arrays=arrays, sparse_matrices=sparse_matrices)
    # save the instrument dictionary into a json file
    save_dict_to_json(info_dict['instrument_info'], os.path.join(result_midi_dir, 'instruments.json'))
    # add a key value pair storing the midi_md5 of the selected midi file if link_to_msd is set True
    if settings['link_to_msd']:
        return (msd_id, {midi_md5: info_dict['midi_info']})
    else:
        return (midi_md5, info_dict['midi_info'])

def main():
    # traverse from dataset root directory and serarch for midi files
    midi_filepaths = []
    for dirpath, _, filenames in os.walk(settings['dataset_path']):
        for filename in filenames:
            if filename.endswith('.mid'):
                midi_filepaths.append(os.path.join(dirpath, filename))
    # parrallelize the converter if in multicore mode
    if settings['multicores'] > 1:
        kv_pairs = joblib.Parallel(n_jobs=settings['multicores'], verbose=5)(
            joblib.delayed(converter)(midi_filepath) for midi_filepath in midi_filepaths)
        # save the midi dict into a json file
        kv_pairs = [kv_pair for kv_pair in kv_pairs if kv_pair is not None]
        midi_dict = {}
        for key in set([kv_pair[0] for kv_pair in kv_pairs]):
            midi_dict[key] = {}
        for kv_pair in kv_pairs:
            midi_dict[kv_pair[0]].update(kv_pair[1])
        save_dict_to_json(midi_dict, os.path.join(settings['result_path'], 'midis.json'))
    else:
        midi_dict = {}
        for midi_filepath in midi_filepaths:
            kv_pair = converter(midi_filepath)
            midi_dict[kv_pair[0]] = kv_pair[1]
        # save the midi dict into a json file
        save_dict_to_json(midi_dict, os.path.join(settings['result_path'], 'midis.json'))

if __name__ == "__main__":
    main()
