import os
import json
import errno
import argparse
from shutil import copyfile
from config import settings

def parse_args():
    """Return parsed command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+', choices=('lpd', 'lpd-5', 'lmd'),
                        help="datasets to be cleansed ('lpd', 'lpd-5' or "
                             "'lmd')")
    args = parser.parse_args()
    return args.datasets

def make_sure_path_exists(path):
    """Create all intermediate-level directories if the given path does not
    exist"""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def midi_filter(midi_info):
    """Return True for qualified midi files and False for unwanted ones"""
    if midi_info['first_beat_time'] > 0.0:
        return False
    elif midi_info['num_time_signature_change'] > 1:
        return False
    elif midi_info['time_signature'] not in settings['time_signatures']:
        return False
    return True

def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

def main():
    """Main function of teh cleanser"""
    datasets = parse_args()

    with open(settings['match_scores']) as infile:
        match_score_dict = json.load(infile)
    with open(settings['midi_dict']) as infile:
        midi_dict = json.load(infile)

    for msd_id in match_score_dict:
        midi_md5s = sorted(match_score_dict[msd_id],
                           key=match_score_dict[msd_id].get, reverse=True)
        for dataset in datasets:
            for midi_md5 in midi_md5s:
                src = os.path.join(settings[dataset]['matched'],
                                   msd_id_to_dirs(msd_id), midi_md5 + '.npz')
                if os.path.isfile(src):
                    if midi_filter(midi_dict[midi_md5]):
                        song_dir = os.path.join(settings[dataset]['cleansed'],
                                                msd_id_to_dirs(msd_id))
                        make_sure_path_exists(song_dir)
                        dst = os.path.join(song_dir, midi_md5 + '.npz')
                        copyfile(src, dst)
                        break

if __name__ == "__main__":
    main()
