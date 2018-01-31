import os.path
import json
import errno
import argparse
from shutil import copyfile
from config import settings

def parse_args():
    """Return parsed command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+', choices=('lpd', 'lpd-5'),
                        help="datasets to be collected ('lpd', 'lpd-5')")
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

def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

def main():
    """Main function of the colllector"""
    datasets = parse_args()
    with open(settings['match_scores']) as infile:
        match_score_dict = json.load(infile)

    for msd_id in match_score_dict:
        for midi_md5 in match_score_dict[msd_id]:
            for dataset in datasets:
                src = os.path.join(settings[dataset]['full'], midi_md5[0],
                                   midi_md5 + '.npz')
                if not os.path.isfile(src):
                    continue
                song_dir = os.path.join(settings[dataset]['matched'],
                                        msd_id_to_dirs(msd_id))
                make_sure_path_exists(song_dir)
                dst = os.path.join(song_dir, midi_md5 + '.npz')
                copyfile(src, dst)

if __name__ == "__main__":
    main()
