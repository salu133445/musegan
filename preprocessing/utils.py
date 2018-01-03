import os
import json
import numpy as np
import scipy.sparse
from config import settings

def make_sure_path_exists(path):
    """Create all intermediate-level directories if the given path not exist"""
    while True:
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                break
            except OSError as err:
                pass

def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

def get_lmd_midi_path(midi_md5, msd_id=None):
    """Given MIDI MD5 (and MSD ID for lmd_matched), return path to the
    corresponding MIDI file."""
    if msd_id is not None:
        return os.path.join(settings['lmd_dir'], msd_id_to_dirs(msd_id), midi_md5 + '.mid')
    else:
        return os.path.join(settings['lmd_dir'], midi_md5[0], midi_md5 + '.mid')

def get_lpd_dir(midi_md5, msd_id=None):
    """Given MIDI MD5 (and MSD ID for lmd_matched), return the corresponding
    lpd directory."""
    if msd_id is not None:
        return os.path.join(settings['lpd_path'], msd_id_to_dirs(msd_id), midi_md5)
    else:
        return os.path.join(settings['lpd_path'], midi_md5[0], midi_md5)

def get_lpd_5_dir(midi_md5, msd_id=None):
    """Given MIDI MD5 (and MSD ID for lmd_matched), return the corresponding
    lpd-5 directory."""
    if msd_id is not None:
        return os.path.join(settings['lpd_5_path'], msd_id_to_dirs(msd_id), midi_md5)
    else:
        return os.path.join(settings['lpd_5_path'], midi_md5[0], midi_md5)

def get_midi_dict():
    """Read the midi dictionary and return it."""
    with open(os.path.join(settings['lpd_path'], 'midis.json')) as infile:
        midi_dict = json.load(infile)
    return midi_dict

def get_piano_rolls(midi_md5, msd_id=None):
    """Given MIDI MD5 (and MSD ID for lmd_matched), return the corresponding
    multi-track piano-roll (np.ndarray)."""
    lpd_dir = get_lpd_dir(midi_md5, msd_id)
    filepath = os.path.join(lpd_dir, 'piano_rolls.npz')
    _, piano_roll_dict = load_npz(filepath)
    piano_roll_list = [piano_roll_dict[str(idx)] for idx in range(len(piano_roll_dict))]
    return np.array(piano_roll_list)

def get_instrument_dict(midi_md5, msd_id=None):
    """Given MIDI MD5 (and MSD ID for lmd_matched), return the corresponding
    instrument dictionary."""
    lpd_dir = get_lpd_dir(midi_md5, msd_id)
    with open(os.path.join(lpd_dir, 'instruments.json')) as infile:
        instrument_dict = json.load(infile)
    return instrument_dict

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
                # embed indices into filenames for future use when loading
                arrays_dict['_'.join([sparse_matrix_name, 'csc_data'])] = csc_matrix.data
                arrays_dict['_'.join([sparse_matrix_name, 'csc_indices'])] = csc_matrix.indices
                arrays_dict['_'.join([sparse_matrix_name, 'csc_indptr'])] = csc_matrix.indptr
                arrays_dict['_'.join([sparse_matrix_name, 'csc_shape'])] = csc_matrix.shape
        else:
            # convert sparse matrices to sparse representations of arrays if any
            for idx, sparse_matrix in enumerate(sparse_matrices):
                csc_matrix = scipy.sparse.csc_matrix(sparse_matrix)
                # embed indices into filenames for future use when loading
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
        # search for non-sparse arrays
        arrays_name = [filename for filename in loaded.files if "_csc_" not in filename]
        arrays = {array_name: loaded[array_name] for array_name in arrays_name}
        # search for csc matrices
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
