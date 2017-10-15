# Herman Dong 2017.6.13
"""
[How to use]
    
[Notes]
    
"""

# Imports
import numpy as np
import pretty_midi
import os
import errno
import json
import scipy.sparse
import midi

# Local path constants
PARSED_DATA_ROOT = '/home/salu133445/NAS/salu133445/midinet/lmd_parsed'
RESULT_PATH = '/home/salu133445/NAS/salu133445/midinet/lmd_parsed/rock_flat_piano_rolls'
# Path to the file midi_dict.json distributed with the LMD
MIDI_DICT_PATH = '/home/salu133445/NAS/salu133445/midinet/lmd_parsed/midi_dict.json'
SUBSET_LIST = '/home/salu133445/NAS/RichardYang/Dataset/Lakh_MIDI_Dataset/subset_id/rock/rock_id'

# Parameters
LAST_BAR_MODE = 'remove'  # 'fill'
THRE_AVG_NOTES_MELODY = 1.1
THRE_AVG_NOTES_RHYTHM = 2.0
THRE_MAX_NOTES_MELODY = 2.0
THRE_PITCH_COMPLEXITY_MELODY = 1.1
THRE_RHYTHM_COMPLEXITY_RHYTHM = 2.0
FILETYPE = 'npz'  # 'csv', 'npz'
MODE = '3D'  # '2D'
KIND = 'matched'

def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

def get_instruments_dict_path(msd_id):
    return os.path.join(PARSED_DATA_ROOT, 'lmd_{}'.format(KIND), msd_id_to_dirs(msd_id), 'instruments.json')

def get_instruments_path(msd_id, midi_md5, instrument_name):
    return os.path.join(PARSED_DATA_ROOT, 'lmd_{}'.format(KIND), msd_id_to_dirs(msd_id), midi_md5, instrument_name + '.npz')

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def my_filter_drop(midi_dict, msd_id, midi_md5):
    # return False for unwanted midi files and True for collected midi files
    if midi_dict[msd_id][midi_md5]['beat start time'] > 0.0:
        return True
    elif not midi_dict[msd_id][midi_md5]['one tsc'] or not midi_dict[msd_id][midi_md5]['all 4/4']:
        return True 
    elif midi_dict[msd_id][midi_md5]['incomplete at start']:
        return True
    else:
        return False

def is_melody(instruments):
    if instrument_dict[instruments]['is drum']:
        return False
    if instrument_dict[instruments]['instrument category'] == 'Bass':
        return False
    if instrument_dict[instruments]["average notes simultaneously"] == 0.0:
        return False
    if instrument_dict[instruments]["average notes simultaneously"] > THRE_AVG_NOTES_MELODY:
        return False
    if instrument_dict[instruments]["max notes simultaneously"] > THRE_MAX_NOTES_MELODY:
        return False
    if instrument_dict[instruments]["pitch complexity"] < THRE_PITCH_COMPLEXITY_MELODY:
        return False
    return True

def is_rhythm(instruments):
    if instrument_dict[instruments]['is drum']:
        return False
    if instrument_dict[instruments]['instrument category'] == 'Bass':
        return True
    if instrument_dict[instruments]["average notes simultaneously"] == 0.0:
        return False
    if instrument_dict[instruments]["average notes simultaneously"] < THRE_AVG_NOTES_RHYTHM:
        return False
    if instrument_dict[instruments]["rhythm complexity"] < THRE_RHYTHM_COMPLEXITY_RHYTHM:
        return False
    return True

def get_bar_piano_roll(msd_id, midi_md5, instrument_name):
    npz_path = get_instruments_path(msd_id, midi_md5, instrument_name)
    piano_roll_sparse = scipy.sparse.load_npz(npz_path)
    piano_roll = piano_roll_sparse.toarray()
    if int(piano_roll.shape[0]%96) is not 0:
        if LAST_BAR_MODE == 'fill':
            piano_roll = np.concatenate((piano_roll, np.zeros((96-piano_roll.shape[0]%96, 128))), axis=0)
        elif LAST_BAR_MODE == 'remove':
            piano_roll = np.delete(piano_roll,  np.s_[-int(piano_roll.shape[0]%96):], axis=0)
    piano_roll = piano_roll.reshape(-1,96,128)
    return piano_roll

def save_flat_piano_roll(piano_roll, postfix):
    filepath = os.path.join(RESULT_PATH, postfix, msd_id + '_' + postfix + '.'+ FILETYPE)
    if FILETYPE == 'npz':  # compressed scipy sparse matrix
        piano_roll = piano_roll.reshape(-1,128)
        sparse_train_data = scipy.sparse.csc_matrix(piano_roll)
        scipy.sparse.save_npz(filepath, sparse_train_data)
    else:
        if MODE == '2D':
            piano_roll = piano_roll.reshape(-1,128)
        elif MODE == '3D':
            if FILETYPE == 'csv':
                np.savetxt(filepath, piano_roll, delimiter=',')
            elif FILETYPE == 'npy':  # uncompressed numpy matrix
                np.save(filepath, piano_roll)
        else:
            print 'Error: Unknown file saving setting'

if __name__ == "__main__":
    # read midi_dict.json into a python dictionary
    with open(MIDI_DICT_PATH) as f:
        midi_dict = json.load(f)
    subset_id_list = []
    with open(SUBSET_LIST) as file:
        for line in file:
            for word in line.replace("[","").replace("]","").replace(",","").split():
                subset_id_list.append(word.strip('"'))
    # allocate empty output array
    train_data = np.zeros((1,96,128))
    make_sure_path_exists(os.path.join(RESULT_PATH, 'flat'))
    make_sure_path_exists(os.path.join(RESULT_PATH, 'melody_flat'))
    make_sure_path_exists(os.path.join(RESULT_PATH, 'rhythm_flat'))
    # iterate over all midi files listed in the subset id list
    for msd_id in subset_id_list:
        for midi_md5 in midi_dict[msd_id]:
            # drop unwanted midi files
            if my_filter_drop(midi_dict, msd_id, midi_md5):
                continue
            # read instrument.json into a python dictionary
            instrument_dict_path = get_instruments_dict_path(msd_id)
            with open(instrument_dict_path) as f:
                instrument_dict = json.load(f)
            # iterate over all instruments to find melody candidates.
            first_instrument = True
            for instrument in instrument_dict:
                if first_instrument:
                    bar_piano_roll = get_bar_piano_roll(msd_id, midi_md5, instrument).astype(float)
                    flat_piano_roll = bar_piano_roll
                    flat_piano_roll_melody = np.zeros_like(flat_piano_roll, dtype=float)
                    flat_piano_roll_rhythm = np.zeros_like(flat_piano_roll, dtype=float)
                    first_instrument = False
                else:
                    bar_piano_roll = get_bar_piano_roll(msd_id, midi_md5, instrument).astype(float)
                    flat_piano_roll = flat_piano_roll + bar_piano_roll
                if is_melody(instrument):
                    flat_piano_roll_melody = flat_piano_roll_melody + bar_piano_roll
                if is_rhythm(instrument):
                    flat_piano_roll_rhythm = flat_piano_roll_rhythm + bar_piano_roll
            save_flat_piano_roll(flat_piano_roll, 'flat')
            save_flat_piano_roll(flat_piano_roll_melody, 'melody_flat')
            save_flat_piano_roll(flat_piano_roll_rhythm, 'rhythm_flat')