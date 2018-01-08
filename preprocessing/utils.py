import os
import errno
import shutil
import json
import numpy as np
import scipy.sparse
from config import settings


def make_sure_path_exists(path):
    """Create all intermediate-level directories if the given path does not
    exist"""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def copy_files_in_dir(src, dst):
    """Copy all files in the source directory to the destination directory.
    Create all intermediate-level directories if destination directory does
    not exist"""
    make_sure_path_exists(dst)
    src_files = os.listdir(src)
    for filename in src_files:
        full_filename = os.path.join(src, filename)
        if os.path.isfile(full_filename):
            shutil.copy(full_filename, dst)

def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

def msd_id_to_h5_file(msd_id):
    """Given an MSD ID, return the path to the corresponding h5 file"""
    return os.path.join(settings['lmd_dir'], 'lmd_matched_h5', msd_id_to_dirs(msd_id) + '.h5')

def get_file_dirs_postfix(midi_md5, msd_id=None):
    """Given MIDI MD5 (and MSD ID for lmd_matched), return the postfix
    directory path."""
    if msd_id is not None:
        return os.path.join(msd_id_to_dirs(msd_id), midi_md5)
    else:
        return os.path.join(midi_md5[0], midi_md5)

def get_lmd_midi_path(midi_md5, msd_id=None, matched=False):
    """Given MIDI MD5 (and MSD ID for lmd_matched), return path to the
    corresponding MIDI file."""
    subset_name = 'lmd_matched' if matched else 'lmd_full'
    return os.path.join(settings['lmd_dir'], subset_name, get_file_dirs_postfix(midi_md5, msd_id) + '.mid')

def get_lpd_subset_dir(matched=False, merged=False, cleansed=False):
    """Return the corresponding lpd subset directory. If the parameter
    'matched' is True, the matched subset is used. If the parameter
    'cleansed' is True, the cleansed subset is used. If the parameter
    'merged' is True, the merged subset is used."""
    if cleansed:
        subset_name = 'lpd_5_cleansed' if merged else 'lpd_cleansed'
    else:
        if matched:
            subset_name = 'lpd_5_matched' if merged else 'lpd_matched'
        else:
            subset_name = 'lpd_5_full' if merged else 'lpd_full'
    return os.path.join(settings['lpd_dir'], subset_name)

def get_lpd_file_dir(midi_md5, msd_id=None, merged=False, cleansed=False):
    """Given MIDI MD5 (and MSD ID for lmd_matched), return the corresponding
    lpd directory. If the parameter 'cleansed' is True, the cleansed subset
    is used. If the parameter 'merged' is True, the merged subset is used."""
    matched = (msd_id is not None)
    lpd_subset_dir = get_lpd_subset_dir(matched, merged, cleansed)
    file_dirs_postfix = get_file_dirs_postfix(midi_md5, msd_id)
    return os.path.join(lpd_subset_dir, file_dirs_postfix)

def get_midi_dict(cleansed=False):
    """Read the midi dictionary and return it. If parameter 'cleansed' is
    True, the cleansed midi_dict will be read. Default is False."""
    lpd_matched_dir = get_lpd_subset_dir(True, False, cleansed)
    lpd_matched_merged_dir = get_lpd_subset_dir(True, True, cleansed)
    midi_dict_path_candidates = [os.path.join(lpd_matched_dir, 'midis.json'),
                                 os.path.join(lpd_matched_merged_dir, 'midis.json')]
    if os.path.isfile(midi_dict_path_candidates[0]):
        with open(midi_dict_path_candidates[0]) as infile:
            midi_dict = json.load(infile)
    elif os.path.isfile(midi_dict_path_candidates[1]):
        with open(midi_dict_path_candidates[1]) as infile:
            midi_dict = json.load(infile)
    else:
        if cleansed:
            raise IOError('Cannot find midi dictionary in lpd-cleansed or lpd-5-cleansed directory.')
        else:
            raise IOError('Cannot find midi dictionary in lpd or lpd-5 directory.')
    return midi_dict

def get_msd_meta_dict():
    """Read the MSD metadata dictionary and return it."""
    with open(settings['msd_meta_filepath']) as infile:
        msd_meta_dict = json.load(infile)
    return msd_meta_dict

def get_match_score_dict():
    """Read the match confidence score dictionary and return it."""
    with open(settings['match_score_filepath']) as infile:
        match_score_dict = json.load(infile)
    return match_score_dict

def get_piano_rolls(midi_md5, msd_id=None, cleansed=False):
    """Given MIDI MD5 (and MSD ID for lmd_matched), return the corresponding
    multi-track piano-roll (np.ndarray)."""
    lpd_dir = get_lpd_file_dir(midi_md5, msd_id, False, cleansed)
    filepath = os.path.join(lpd_dir, 'piano_rolls.npz')
    _, piano_roll_dict = load_npz(filepath)
    piano_roll_list = [piano_roll_dict[str(idx)] for idx in range(len(piano_roll_dict))]
    return np.array(piano_roll_list)

def get_instrument_dict(midi_md5, msd_id=None, merged=False, cleansed=False):
    """Given MIDI MD5 (and MSD ID for lmd_matched), return the corresponding
    instrument dictionary."""
    lpd_dir = get_lpd_file_dir(midi_md5, msd_id, merged, cleansed)
    with open(os.path.join(lpd_dir, 'instruments.json')) as infile:
        instrument_dict = json.load(infile)
    return instrument_dict

def save_dict_to_json(data, filepath):
    """Save dictionary to a json file."""
    with open(filepath, 'w') as outfile:
        json.dump(data, outfile)

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
