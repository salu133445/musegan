from __future__ import division
from __future__ import print_function

import scipy.sparse
import json
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
import scipy.io as sio
import librosa
import os

def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix. E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join( msd_id[2], msd_id[3], msd_id[4], msd_id)
def csc_to_array(csc):
    return scipy.sparse.csc_matrix((csc['data'], csc['indices'], csc['indptr']), shape= csc['shape']).toarray()
def reshape_to_bar(flat_array):
    return flat_array.reshape(-1,96,128)
def is_empty_bar(bar):
    return not np.sum(bar)

ROOT_TRACKS = 'tracks'
PATH_PIANO_ROLL = join(ROOT_TRACKS, 'Piano_Roll')
prefix = ['Bass', 'Drum', 'Guitar', 'Other', 'Piano']
PATH_INSTRU_ACT = join(ROOT_TRACKS, 'act_instr')
PATH_ALL_ACT = join(ROOT_TRACKS, 'act_all')

if __name__ == "__main__":
    
    if not os.path.exists(PATH_INSTRU_ACT):
        os.makedirs(PATH_INSTRU_ACT)
    if not os.path.exists(PATH_ALL_ACT ):
        os.makedirs(PATH_ALL_ACT )
    if not os.path.exists(PATH_PIANO_ROLL):
        os.makedirs(PATH_PIANO_ROLL)

    
    song_list = onlyfiles = [f.split('.')[0] for f in listdir(join(ROOT_TRACKS, 'Drum')) if isfile(join(join(ROOT_TRACKS, 'Drum'), f))]

    thres = 3
    numOfSong = len(song_list)
    for song_idx in range(numOfSong ):
        
        msd_id = song_list[song_idx]
        sys.stdout.write('{0}/{1}\r'.format(song_idx, numOfSong))
        sys.stdout.flush()
        
        song_piano_rolls = []
        list_is_empty = []

        piano_roll = reshape_to_bar(csc_to_array(np.load(join(ROOT_TRACKS,prefix[0], msd_id+'.npz'))))
        song_piano_rolls.append(piano_roll)

        for idx in range(1,5):
            piano_roll_tmp = reshape_to_bar(csc_to_array(np.load(join(ROOT_TRACKS,prefix[idx], msd_id+'.npz'))))
            piano_roll += piano_roll_tmp
            song_piano_rolls.append(piano_roll_tmp)

        piano_roll = np.concatenate((piano_roll[:]))
        piano_roll = piano_roll.T

        numOfBar = song_piano_rolls[0].shape[0]
        instr_act = np.zeros((numOfBar,5))
        all_act = np.zeros(numOfBar)
        chroma = np.zeros_like(song_piano_rolls[0])

        for bar_idx in range(numOfBar):
            for pre_idx in range(5):
                bar = song_piano_rolls[pre_idx][bar_idx,:,:]
                instr_act[bar_idx, pre_idx] = not is_empty_bar(bar)
                all_act[bar_idx] = np.sum(instr_act[bar_idx, :]) >= thres

        sio.savemat(os.path.join(PATH_PIANO_ROLL, msd_id+'.mat'), {'piano_roll':piano_roll})
        np.save(join(PATH_INSTRU_ACT, msd_id+'.npy'), instr_act)
        np.save(join(PATH_ALL_ACT, msd_id+'.npy'), all_act)