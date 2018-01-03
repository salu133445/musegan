from __future__ import print_function
import os
from shutil import copyfile
import numpy as np
from config import settings
from utils import get_lpd_dir, get_lpd_5_dir, make_sure_path_exists
from utils import get_piano_rolls, get_instrument_dict, get_midi_dict
from utils import save_npz

if settings['multicore'] > 1:
    import joblib

def merge_piano_rolls(piano_rolls):
    """Given a multi-track piano-roll (np.ndarray), return the boolean sum
    along the pitch axis."""
    if piano_rolls.ndim < 3:
        return piano_rolls
    else:
        return np.sum((piano_rolls > 0), axis=2, dtype=bool)

def merger(midi_md5, msd_id=None):
    """Given msd_id and midi_md5, merge the multi-track piano-rolls into five
    tracks (bass, drums, guitar, piano and strings). The merged piano-rolls
    are saved and the related files are copied to the result directory."""
    # load piano-rolls and instrument dictionary
    piano_rolls = get_piano_rolls(midi_md5, msd_id)
    instrument_dict = get_instrument_dict(midi_md5, msd_id)
    # categorize piano-rolls into five lists
    category_list = {'bass': [], 'drums': [], 'guitar': [], 'piano': [], 'strings': []}
    for idx in range(piano_rolls.shape[0]):
        if instrument_dict[str(idx)]['is_drum']:
            category_list['drums'].append(idx)
        elif instrument_dict[str(idx)]['family_num'] == 0:
            category_list['piano'].append(idx)
        elif instrument_dict[str(idx)]['family_num'] == 3:
            category_list['guitar'].append(idx)
        elif instrument_dict[str(idx)]['family_num'] == 4:
            category_list['bass'].append(idx)
        else:
            category_list['strings'].append(idx)
    # merge piano-rolls in the same category into one piano-roll
    category_piano_rolls = {}
    for category in category_list:
        to_be_merged = piano_rolls[category_list[category]]
        category_piano_rolls[category] = merge_piano_rolls(to_be_merged)
    # save merged piano-rolls
    lpd_5_dir = get_lpd_5_dir(midi_md5, msd_id)
    make_sure_path_exists(lpd_5_dir)
    for category in category_piano_rolls:
        save_npz(os.path.join(lpd_5_dir, category+'.npz'))
    # copy other files to the result directory
    src_dir = get_lpd_dir(midi_md5, msd_id)
    for filename in ['arrays.npz', 'instruments.json']:
        copyfile(os.path.join(src_dir, filename), os.path.join(lpd_5_dir, filename))
        copyfile(os.path.join(src_dir, filename), os.path.join(lpd_5_dir, filename))

def main():
    # open the midi dictionary
    midi_dict = get_midi_dict()
    # parallelize the converter if in multicore mode
    if settings['multicore'] > 1:
        if settings['dataset'] == 'matched':
            joblib.Parallel(n_jobs=settings['multicore'], verbose=5)(
                joblib.delayed(merger)(midi_md5, msd_id) for msd_id in midi_dict for midi_md5 in midi_dict[msd_id])
        else:
            joblib.Parallel(n_jobs=settings['multicore'], verbose=5)(
                joblib.delayed(merger)(midi_md5) for midi_md5 in midi_dict)
    else:
        if settings['dataset'] == 'matched':
            for msd_id in midi_dict:
                for midi_md5 in midi_dict[msd_id]:
                    merger(midi_md5, msd_id)
        else:
            for midi_md5 in midi_dict:
                merger(midi_md5)

if __name__ == "__main__":
    main()
