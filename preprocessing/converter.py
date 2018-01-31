from __future__ import print_function
import os
import json
import errno
import argparse
import warnings
from pypianoroll import Multitrack, Track
from config import settings
if settings['multicore'] > 1:
    import joblib

warnings.filterwarnings('ignore')

def parse_args():
    """Return parsed command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+', choices=('lpd', 'lpd-5'),
                        help="convert to normal or merged multi-track "
                             "piano-rolls or both ('lpd', 'lpd-5')")
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

def change_prefix(path, src, dst):
    """Return the path with its prefix changed from `src` to `dst`"""
    return os.path.join(dst, os.path.relpath(path, src))

def get_midi_path(root):
    """Return a list of paths to MIDI files in `root` (recursively)"""
    filepaths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.mid'):
                filepaths.append(os.path.join(dirpath, filename))
    return filepaths

def get_midi_info(pm):
    """Return useful information from a pretty_midi.PrettyMIDI instance"""
    if pm.time_signature_changes:
        pm.time_signature_changes.sort(key=lambda x: x.time)
        first_beat_time = pm.time_signature_changes[0].time
    else:
        first_beat_time = pm.estimate_beat_start()

    tc_times, tempi = pm.get_tempo_changes()

    if len(pm.time_signature_changes) == 1:
        time_sign = '{}/{}'.format(pm.time_signature_changes[0].numerator,
                                    pm.time_signature_changes[0].denominator)
    else:
        time_sign = None

    midi_info = {
        'first_beat_time': first_beat_time,
        'num_time_signature_change': len(pm.time_signature_changes),
        'time_signature': time_sign,
        'tempo': tempi[0] if len(tc_times) == 1 else None
    }

    return midi_info

def get_merged(multitrack):
    """Return a `pypianoroll.Multitrack` instance with piano-rolls merged to
    five tracks (Bass, Drums, Guitar, Piano and Strings)"""
    category_list = {'Bass': [], 'Drums': [], 'Guitar': [], 'Piano': [],
                     'Strings': []}
    for idx, track in enumerate(multitrack.tracks):
        if track.is_drum:
            category_list['Drums'].append(idx)
        elif track.program//8 == 0:
            category_list['Piano'].append(idx)
        elif track.program//8 == 3:
            category_list['Guitar'].append(idx)
        elif track.program//8 == 4:
            category_list['Bass'].append(idx)
        else:
            category_list['Strings'].append(idx)
    program_dict = {'Piano': 0, 'Drums': 0, 'Guitar': 24, 'Bass': 32,
                    'Strings': 48}
    tracks = []
    for key in category_list:
        if category_list[key]:
            merged = multitrack[category_list[key]].get_merged_pianoroll()
            tracks.append(Track(merged, program_dict[key], key=='Drums', key))
        else:
            tracks.append(Track(None, program_dict[key], key=='Drums', key))
    return Multitrack(None, tracks, multitrack.tempo, multitrack.downbeat,
                      multitrack.beat_resolution, multitrack.name)

def converter(filepath, datasets, midi_dict):
    """Save a multi-track piano-roll converted from a MIDI file to target
    dataset directory and update MIDI information to `midi_dict`"""
    try:
        midi_md5 = os.path.splitext(os.path.basename(filepath))[0]
        multitrack = Multitrack(beat_resolution=settings['beat_resolution'],
                                name=midi_md5)

        pm = pretty_midi.PrettyMIDI(filepath)

        midi_info = get_midi_info(pm)

        multitrack.parse_pretty_midi(pm)

        if 'lpd' in datasets:
            dst = change_prefix(os.path.dirname(filepath),
                                settings['lmd']['full'],
                                settings['lpd']['full'])
            make_sure_path_exists(dst)
            multitrack.save(os.path.join(dst, midi_md5 + '.npz'))

        if 'lpd-5' in datasets:
            dst = change_prefix(os.path.dirname(filepath),
                                settings['lmd']['full'],
                                settings['lpd-5']['full'])
            merged = get_merged(multitrack)
            make_sure_path_exists(dst)
            merged.save(os.path.join(dst, midi_md5 + '.npz'))

        return (midi_md5, midi_info)

    except:
        return None

def main():
    """Main function of the converter"""
    datasets = parse_args()
    midi_paths = get_midi_path(settings['lmd']['full'])
    midi_dict = {}

    if settings['multicore'] > 1:
        kv_pairs = joblib.Parallel(n_jobs=settings['multicore'], verbose=5)(
            joblib.delayed(converter)(midi_path, datasets, midi_dict)
            for midi_path in midi_paths)
        for kv_pair in kv_pairs:
            if kv_pair is not None:
                midi_dict[kv_pair[0]] = kv_pair[1]
    else:
        counter = 0
        for midi_path in midi_paths:
            kv_pair = converter(midi_path, datasets, midi_dict)
            if kv_pair is not None:
                midi_dict[kv_pair[0]] = kv_pair[1]

    with open(os.path.join(settings['midi_dict']), 'w') as outfile:
        json.dump(midi_dict, outfile)

    print("[Done] {} files out of {} have been successfully converted".format(
        len(midi_dict), len(midi_paths)))

if __name__ == "__main__":
    main()
