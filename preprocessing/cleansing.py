from __future__ import print_function
import os
import joblib
from utils import copy_files_in_dir, save_dict_to_json
from utils import get_lpd_file_dir, get_lpd_subset_dir
from utils import get_midi_dict, get_match_score_dict
from config import settings

def midi_filter(midi_dict, midi_md5, msd_id):
    """Return True for qualified midi files and False for unwanted ones"""
    if midi_dict[msd_id][midi_md5]['beat_start_time'] > 0.0:
        return False
    elif midi_dict[msd_id][midi_md5]['num_time_signature_changes'] > 1:
        return False
    elif midi_dict[msd_id][midi_md5]['time_signature'] not in settings['cleansing']['time_signatures']:
        return False
    elif midi_dict[msd_id][midi_md5]['incomplete_at_start']:
        return False
    return True

def cleanser(midi_dict, match_score_dict, msd_id):
    # find the midi file with the highest match score
    midi_md5 = max(match_score_dict[msd_id], key=match_score_dict[msd_id].get)
    if midi_filter(midi_dict, midi_md5, msd_id):
        src_dir = get_lpd_file_dir(midi_md5, msd_id, settings['cleansing']['use_merged_dataset'], False)
        dst_dir = get_lpd_file_dir(midi_md5, msd_id, settings['cleansing']['use_merged_dataset'], True)
        #copy_files_in_dir(src_dir, dst_dir)
        return (msd_id, {'midi_md5': midi_md5, 'midi_info': midi_dict[msd_id][midi_md5]})
    return None

def main():
    # read the midi dictionary
    midi_dict = get_midi_dict()
    match_score_dict = get_match_score_dict()
    #  parallelize the cleanser if in multicore mode
    if settings['multicore'] > 1:
        kv_pairs = joblib.Parallel(n_jobs=settings['multicore'], verbose=5)(
            joblib.delayed(cleanser)(midi_dict, match_score_dict, msd_id) for msd_id in midi_dict)
        # save the cleansed midi dict into a json file
        kv_pairs = [kv_pair for kv_pair in kv_pairs if kv_pair is not None]
        cleansed_midi_dict = {}
        for key in set([kv_pair[0] for kv_pair in kv_pairs]):
            cleansed_midi_dict[key] = {}
        for kv_pair in kv_pairs:
            cleansed_midi_dict[kv_pair[0]].update(kv_pair[1])
    else:
        cleansed_midi_dict = {}
        for msd_id in midi_dict:
            kv_pair = cleanser(midi_dict, match_score_dict, msd_id)
            if kv_pair is not None:
                cleansed_midi_dict[kv_pair[0]] = kv_pair[1]
    # save the cleansed midi dict into a json file
    dst_dir = get_lpd_subset_dir(True, settings['cleansing']['use_merged_dataset'], True)
    save_dict_to_json(cleansed_midi_dict, os.path.join(dst_dir, 'midis.json'))

if __name__ == "__main__":
    main()
