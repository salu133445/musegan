'''
wayne

Directly parse data of trakcs in lmd_mathced. The id list if from Rock_C
For type: BASS (B), DRUM (D), PIANO (P), OTHER (O), GUITAR (G)
'''
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import os
import librosa
import errno
import json
import scipy.sparse


ROOT = '/home/wayne/NAS' #'Z:' 
KIND = 'matched'
DATA_SET_ROOT = os.path.join(ROOT, 'salu133445/midinet/lmd_parsed')
MIDI_DICT_PATH = os.path.join(ROOT, 'salu133445/midinet/lmd_parsed/midi_dict.json')
SUBSET_LIST = 'rock_C_id'
RESULT_PATH = 'tracks'
LAST_BAR_MODE = 'remove'
FILETYPE = 'npz'  # 'csv', 'npz'

def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix. E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join( msd_id[2], msd_id[3], msd_id[4], msd_id)
def csc_to_array(csc):
    return scipy.sparse.csc_matrix((csc['data'], csc['indices'], csc['indptr']), shape= csc['shape']).toarray()

def get_instruments_dict_path(msd_id):
    return os.path.join(DATA_SET_ROOT, 'lmd_{}'.format(KIND), msd_id_to_dirs(msd_id), 'instruments.json')

def get_instruments_path(msd_id, midi_md5, instrument_name):
    return os.path.join(DATA_SET_ROOT, 'lmd_{}'.format(KIND), msd_id_to_dirs(msd_id), midi_md5, instrument_name + '.npz')

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
    filepath = os.path.join(RESULT_PATH, postfix, msd_id + '.'+ FILETYPE)
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
            print( 'Error: Unknown file saving setting')

if __name__ == "__main__":
    with open(MIDI_DICT_PATH) as f:
            midi_dict = json.load(f)
            
    subset_id_list = []
    with open(SUBSET_LIST) as file:
        for line in file:
            for word in line.replace("[","").replace("]","").replace(",","").split():
                subset_id_list.append(word.strip('"'))
    print(len(subset_id_list )) # 6646 songs in rock_C_id


    counter = 0
    for song_idx in range(len(subset_id_list)):
        msd_id = subset_id_list[song_idx]
        prefix = ['Bass', 'Drum', 'Guitar', 'Other', 'Piano']
        flag_act = [0,0,0,0,0]

        # make dirs
        if not os.path.exists(RESULT_PATH):
            os.makedirs(RESULT_PATH)

        for p in prefix:
            instr_dir = os.path.join(RESULT_PATH, p)
            if not os.path.exists(instr_dir):
                os.makedirs(instr_dir)

        # save data
        for midi_md5 in midi_dict[msd_id]:
            if my_filter_drop(midi_dict, msd_id, midi_md5):
                print(song_idx, 'parsing error')
                continue

            ## check parsing error
            instrument_dict_path = get_instruments_dict_path(msd_id)
            with open(instrument_dict_path) as f:
                instrument_dict = json.load(f)

            ## check instrument error
            instrus_act = list(set([key.split('_')[0] for key in instrument_dict.keys()]))
         
            for p in range(5):
                if(prefix[p]) in instrus_act:
                    flag_act[p] = 1

            if sum(flag_act) is not 5:
                print(song_idx, 'instr error',  instrus_act)
                continue

            ## get instructment info

            ## flatten
            # Strategy:
            # Compress each of (Bass, Drum, Guitar, Piano) to 1
            # Compress all of Others (include string) to 1

            instrus = [key for key in instrument_dict.keys()]
            type_list = [[],[],[],[],[]]
           
            for i in instrus:
                info = i.split('_')
                try:
                    prefix_idx = prefix.index(info[0])
                    type_list[prefix_idx].append(i)
                except:
                    print(info[0])
                    prefix_idx = prefix.index('Other')
                    type_list[prefix_idx].append(i)
                
               

            bar_remplate = get_bar_piano_roll(msd_id, midi_md5, type_list[0][0]).astype(float)

            for idx in range(5):
                piano_roll = np.zeros_like( bar_remplate, dtype=float)

                for tt in type_list[idx]:
                    piano_roll += get_bar_piano_roll(msd_id, midi_md5, tt).astype(float)

                # plt.figure()
                # librosa.display.specshow(piano_roll[80,:,:].T, y_axis='cqt_note', cmap=plt.cm.hot)
                # plt.title(prefix[idx])
                # print(piano_roll.shape) # (nbar, 96, 128)
                save_flat_piano_roll(piano_roll, prefix[idx])
            
            counter += 1
            print('%d/%d' %( counter, song_idx), 'OK!', instrus_act)
