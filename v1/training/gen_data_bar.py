from __future__ import division
from __future__ import print_function




import os
import numpy as np
from os import listdir
from os.path import isfile, join
import sys
import scipy.io as sio
import scipy.sparse
import json

PATH_SEG = 'structure/labs'
ROOT_TRACKS = 'tracks'
PATH_INSTRU_ACT = join(ROOT_TRACKS, 'act_instr')
PATH_SAVE_BAR = 'data_bar'
PATH_SAVE_PHR = 'data_phr'
TRA_SIZE = 2000
VAL_SIZE = 800
SAMPLE_RATIO = 0.5


prefix = ['Bass', 'Drum', 'Guitar', 'Other', 'Piano', 'instr_act']
PATH_SEG = 'structure/labs'
song_list = onlyfiles = [f.split('.')[0] for f in listdir(PATH_SEG) if isfile(join(PATH_SEG, f))]

def csc_to_array(csc):
    return scipy.sparse.csc_matrix((csc['data'], csc['indices'], csc['indptr']), shape= csc['shape']).toarray()
def reshape_to_bar(flat_array):
    return flat_array.reshape(-1,96,128)

def save_phrases(rnd_list, save_dir='tr'):
    ### Strategy #### ################
    # (numOfbar of a phrase)
    # (>= 8)
    #         % 4 = 0
    #             = 1 ed-1
    #             = 2 st,ed-1
    #             = 3 ed+1
    #      4n + 8, n =0,1,2,3
    #
    # (<= 8)
    #             = 7 ed+1
    #             = 6 st,ed+1
    ##################################

    path_tosave = join( PATH_SAVE_PHR, save_dir)
    if not os.path.exists(path_tosave):
        os.makedirs(path_tosave)

    barsOfphrase_list = [[],[],[],[],[],[]]
    for song_idx in range(len(rnd_list)):
        msd_id = song_list[tra_song_idx[song_idx]]
        sys.stdout.write('{0}/{1}\r'.format(song_idx, len(rnd_list)))
        sys.stdout.flush()
        f = open(join(PATH_SEG, msd_id+'.lab'),'r')

        info_list = []
        for line in f.readlines():
            line = line.strip().split(' ')
            st, ed, lab,  = int(float(line[0])), int(float(line[1])), line[2]
            info_list.append((st, ed, lab, ed-st))
        tmp_phr_num = len(info_list)

        parsed_phrase_list = []
        song_tracks = []
        for pre_idx in range(5):
            song_tracks.append(reshape_to_bar(csc_to_array(np.load(join(ROOT_TRACKS, prefix[pre_idx], msd_id+'.npz')))))
        act_instr = np.load(join(PATH_INSTRU_ACT, msd_id+'.npy'))
        for pidx in range(1, tmp_phr_num-1):
            st, ed, phr_len = info_list[pidx ][0], info_list[pidx ][1], info_list[pidx ][3]
            if(phr_len >=8):
                if(phr_len%4==1):
                    ed -= 1
                if(phr_len%4==2):
                    st += 1
                    ed -= 1
                if(phr_len%4==3):
                    ed = ed+1
                phr_len = ed - st
                limit = phr_len - 7
                for t in range(0, limit, 4):
                    parsed_phrase_list.append((st+t,st+t+8))

            else:
                if(phr_len == 7):
                    parsed_phrase_list.append((st, ed+1))
                if(phr_len == 6):
                    parsed_phrase_list.append((st-1, ed+1))
                else:
                    pass


        for pidx in range(len(parsed_phrase_list)):
            for pre_idx in range(5):
                st, ed= parsed_phrase_list[pidx]
                barsOfphrase_list[pre_idx].append(song_tracks[pre_idx][st:ed, : ,:].astype(bool))
            barsOfphrase_list[5].append(act_instr[st:ed,:].astype(bool))

        # print(parsed_phrase_list, len(parsed_phrase_list))
        f.close()
    print('[*] Saving...')
    print('\n')

    for idx in range(6):
        track_list = barsOfphrase_list[idx]
        tmp = np.asarray(track_list)
        print( tmp.shape, tmp.dtype)
        np.save(join(path_tosave, prefix[idx]+'.npy'), tmp)

def save_bars(rnd_list, save_dir='tr'):
    total_bar_len = 0
    path_tosave = join( PATH_SAVE_BAR, save_dir)
    if not os.path.exists(path_tosave):
        os.makedirs(path_tosave)

    bar_list = [[],[],[],[],[], []] # len = 6, 'Bass', 'Drum', 'Guitar', 'Other', 'Piano', 'instr_act'
    for song_idx in range(len(rnd_list)):
        msd_id = song_list[tra_song_idx[song_idx]]
        sys.stdout.write('{0}/{1}\r'.format(song_idx, len(rnd_list)))
        sys.stdout.flush()
        f = open(join(PATH_SEG, msd_id+'.lab'),'r')

        info_list = []
        unique_label = []
        bar_list_song = []
        for line in f.readlines():
            line = line.strip().split(' ')
            st, ed, lab,  = int(float(line[0])), int(float(line[1])), line[2]
            info_list.append((st, ed, lab, ed-st))
            unique_label.append(lab)

        ### Get bar id list of one song ###
        unique_label  = list(set(unique_label ))
        flag = np.zeros(len(unique_label))
        for pi in range(len(info_list)-2,0,-1):
            lab = info_list[pi][2]
            pos = unique_label.index(lab)

            if not flag[pos]:
                flag[pos] += 1
                bar_list_song.extend([x for x in range(info_list[pi][0],info_list[pi][1])])
            else:
                sample_phr = int(info_list[pi][3] * SAMPLE_RATIO)
                tmp_rnd_list = np.random.permutation(info_list[pi][3])
                cand_bar_idx = tmp_rnd_list[:sample_phr]
                bar_list_song.extend([x+info_list[pi][0] for x in cand_bar_idx])
                flag[pos] += 1

        ### Generate data by bar id list ###
        act_instr = np.load(join(PATH_INSTRU_ACT, msd_id+'.npy'))
        total_bar_len += act_instr.shape[0]
        song_tracks = []
        for pre_idx in range(5):
            song_tracks.append(reshape_to_bar(csc_to_array(np.load(join(ROOT_TRACKS, prefix[pre_idx], msd_id+'.npz')))))

        for idx in range(len(bar_list_song)):
            bar_idx = bar_list_song[idx]
            for pre_idx in range(5):
                bar_list[pre_idx].append(song_tracks[pre_idx][bar_idx,:,:].astype(bool))
            bar_list[5].append(act_instr[bar_idx,:].astype(bool))

        f.close()

    # print(bar_list)

    # print(num_val_phr, num_val_phr_bar, num_val_bar)
    print('%d Songs Selected' % (len(rnd_list)))
    print('%d bars selected in %d' % (len(bar_list[0]), total_bar_len))
    print('[*] Saving...')
    print('\n')

    for idx in range(6):
        track_list = bar_list[idx]
        tmp = np.asarray(track_list)
        print( tmp.shape, tmp.dtype)
        np.save(join(path_tosave, prefix[idx]+'.npy'), tmp)

if __name__ == "__main__":



    numOfSong = len(song_list)
    rnd_list = list(np.random.permutation(numOfSong ))
    tra_song_idx = rnd_list[:TRA_SIZE]
    val_song_idx = rnd_list[TRA_SIZE:TRA_SIZE+VAL_SIZE]
    test_song_idx = rnd_list[TRA_SIZE+VAL_SIZE:]
    total_bar_len = 0

    # Saving
    save_bars(test_song_idx, save_dir='test')
    save_phrases(test_song_idx, save_dir='test')
    save_bars(tra_song_idx, save_dir='tra')
    save_phrases(tra_song_idx, save_dir='tra')
    save_bars(val_song_idx, save_dir='val')
    save_phrases(val_song_idx, save_dir='val')

    # Export random list
    out_list = {'tra_song_idx': tra_song_idx, 'val_song_idx': val_song_idx,'test_song_idx':test_song_idx}
    with open('rnd_idx_list', 'w') as outfile:
        json.dump(out_list, outfile)