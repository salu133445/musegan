# Herman Dong 2017.5.18 v1
# Last modified: Herman Dong 2017.09.27 v2
"""
[How to use]
    (1) set the correct DATASET_ROOT, SEARCH_ROOT, RESULTS_PATH constant
    (2) set the correct SCORE_FILE path constant
    (3) set the desired FILETYPE constant
    (4) set the correct CONFIDENCE_TH, BEAT_RESOLUTION constant
    (5) run the following
            python2 midi2pianoroll.py
[Notes]
    Only the midi file with highest confidence score is chosen.
    occurrence ratio: the ratio of the number of bars that the instrument shows up to the number of bars in total
    average notes simultaneously: the average number of notes occurs simultaneously
    max notes simultaneously: the maximum number of notes occurs simultaneously
    rhythm complexity: average number of notes(onset only) in a bar
    pitch complexity: average number of notes with different pitches
"""

from __future__ import print_function
import os
import shutil
import json
import numpy as np
import pretty_midi
from config import settings

if settings['filetype'] == 'npz':
    import scipy.sparse

# # Local path constants
# DATASET_ROOT = '/home/salu133445/NAS/RichardYang/Dataset/Lakh_MIDI_Dataset'
# SEARCH_ROOT = '/home/salu133445/NAS/RichardYang/Dataset/Lakh_MIDI_Dataset/lmd_matched'
# RESULTS_PATH = '/home/salu133445/NAS/salu133445/midinet/lmd_parsed'
# # Path to the file match_scores.json distributed with the LMD
# SCORE_FILE = '/home/salu133445/NAS/RichardYang/Dataset/Lakh_MIDI_Dataset/match_scores.json'

# # Parameters
# CONFIDENCE_TH = 0.3
# BEAT_RESOLUTION = 24
# FILETYPE = 'npz'  # 'npy', 'csv'
# KIND = 'matched'  # 'aligned'

def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

def get_midi_path(msd_id, midi_md5):
    """Given an MSD ID and MIDI MD5, return path to a MIDI file.
    kind should be one of 'matched' or 'aligned'. """
    return os.path.join(settings['dataset_path'], msd_id_to_dirs(msd_id), midi_md5 + '.mid')

# def instrument_to_family(instrument):
#     """Given an instrument object, return its (instrument) family[string]."""
#     if instrument.is_drum:
#         return 'drums'
#     elif instrument.program+1 < 9:
#         return 'piano'
#     elif instrument.program+1 > 24 and instrument.program+1 < 33:
#         return 'guitar'
#     elif instrument.program+1 > 32 and instrument.program+1 < 41:
#         return 'bass'
#     elif instrument.program+1 > 40 and instrument.program+1 < 49:
#         return 'strings'
#     else:
#         return 'other'

def get_time_signature_changes_info(pm):
    """Given an pretty_midi object, return its time signature changes info[dict]."""
    # create tsc_info dictionary, time_signature is defaulted to None
    tsc_info = {'num_tsc': len(pm.time_signature_changes),
                'time_signature': None}
    # set time_signature when there is only one time signature change
    if len(pm.time_signature_changes) == 1:
        tsc_info['time_signature'] = pm.time_signature_changes[0].denominator + '/' + \
                                     pm.time_signature_changes[0].numerator
    return tsc_info

def get_beats_info_and_arrays(pm):
    """Given an pretty_midi object, return its beats_info and beats arrays."""
    # use built-in method in pretty_midi to get beat_start_time, beats and downbeats
    beat_start_time = pm.time_signature_changes[0].time if pm.time_signature_changes else 0.0
    beats = pm.get_beats(beat_start_time)
    downbeats = pm.get_downbeats(beat_start_time)
    # set number of beats and downbeats to beats info
    num_beats = len(beats)
    incomplete_at_start = (downbeats[0] > beats[0])
    num_bars = len(downbeats) + int(incomplete_at_start)
    # create an empty beats array and an empty downbeats array
    beats_array = np.zeros(shape=(settings['beat_resolution']*num_beats, 1), dtype=bool)
    downbeats_array = np.zeros(shape=(settings['beat_resolution']*num_beats, 1), dtype=bool)
    # fill in the beats array and the downbeats array
    beats_array[0:-1:settings['beat_resolution']] = True
    for downbeat, idx in enumerate(downbeats):
        idx_to_fill = np.searchsorted(beats, downbeat, side='right')
        downbeats_array[idx_to_fill] = True
    # create dictionary for beats info and beats arrays
    beats_info = {'beat_start_time': beat_start_time,
                  'num_beats': num_beats,
                  'num_bars': num_bars,
                  'incomplete_at_start': incomplete_at_start}
    beats_arrays = {'beats': beats,
                    'downbeats': downbeats,
                    'beats_array': beats_array,
                    'downbeats_array': downbeats_array}
    return beats_info, beats_arrays

def get_tempo_array(pm, beats=None):
    if beats is None:
        beat_start_time = pm.time_signature_changes[0].time if pm.time_signature_changes else 0.0
        beats = pm.get_beats(beat_start_time)
    # create an empty tempo_array
    tempo_array = np.zeros(shape=(settings['beat_resolution']*len(beats), 1))
    # use built-in method in pretty_midi to get tempo change events
    tempo_change_times, tempi = pm.get_tempo_changes()
    if not tempo_change_times:
        # set to default tempo value when no tempo change events
        tempo_array[:] = settings['default_tempo']
    else:
        # deal with the first tempo change event
        tempo_end_beat = (np.searchsorted(beats, tempo_change_times[0], side='right'))
        tempo_array[0:tempo_end_beat] = tempi[0]
        tempo_start_beat = tempo_end_beat

        # deal with the rest tempo change event
        tempo_id = 0
        while tempo_id+1 < len(tempo_change_times):
            tempo_end_beat = (np.searchsorted(beats, tempo_change_times[tempo_id+1], side='right'))
            tempo_array[tempo_start_beat:tempo_end_beat] = tempi[tempo_id]
            tempo_start_beat = tempo_end_beat
            tempo_id += 1
        # deal with the rest beats
        tempo_array[tempo_start_beat:] = tempi[tempo_id]
    return tempo_array

def get_piano_roll(pm, instrument, beats=None, tempo_array=None):
    if beats is None:
        beat_start_time = pm.time_signature_changes[0].time if pm.time_signature_changes else 0.0
        beats = pm.get_beats(beat_start_time)
        num_beats = len(beats)
    if tempo_array is None:
        tempo_array = get_tempo_array(pm, beats)
    # create the piano roll and the onset array
    if settings['velocity'] == 'bool':
        piano_roll = np.zeros(shape=(settings['beat_resolution']*num_beats, 128), dtype=bool)
    elif settings['velocity'] == 'int':
        piano_roll = np.zeros(shape=(settings['beat_resolution']*num_beats, 128), dtype=int)
    onset_array = np.zeros(shape=(settings['beat_resolution']*num_beats, 1), dtype=bool)
    # calculate pixel per beat
    ppbeat = settings['beat_resolution']
    hppbeat = settings['beat_resolution']/2
    # iterate through all the notes in the instrument track
    if not instrument.is_drum:
        for note in instrument.notes:
            if note.end < beats[0]:
                continue
            else:
                # find the corresponding index of the note on/off event
                if note.start >= beats[0]:
                    start_beat = np.searchsorted(beats, note.start, side='right') - 1
                    start_loc = (note.start - beats[start_beat])
                    start_loc = start_loc * pm.tempo_array[int(start_beat*hppbeat)] / 60.0
                else:
                    start_beat = 0
                    start_loc = 0.0
                end_beat = np.searchsorted(beats, note.end, side='right') - 1
                end_loc = (note.end - beats[end_beat])
                end_loc = end_loc * pm.tempo_array[int(end_beat*hppbeat)] / 60.0
                start_idx = int(start_beat*ppbeat + start_loc*hppbeat)
                end_idx = int(end_beat*ppbeat + end_loc*hppbeat)
                # make sure the note length is larger than minimum note length
                if end_idx - start_idx < 2:
                    end_idx = start_idx + 2
                # set values to the pianoroll matrix and the onset array
                if settings['velocity'] == 'bool':
                    piano_roll[start_idx:(end_idx-1), note.pitch] = True
                elif settings['velocity'] == 'int':
                    piano_roll[start_idx:(end_idx-1), note.pitch] = note.velocity
                if start_idx < onset_array.shape[0]:
                    onset_array[start_idx, 0] = True
    else: # drums track
        for note in instrument.notes:
            if note.end < beats[0]:
                continue
            else:
                # find the corresponding index of the note on event (for drums, only note on events are captured)
                if note.start >= beats[0]:
                    start_beat = np.searchsorted(beats, note.start, side='right') - 1
                    start_loc = (note.start - beats[start_beat])
                    start_loc = start_loc * pm.tempo_array[int(start_beat*ppbeat)] / 60.0
                else:
                    start_beat = 0
                    start_loc = 0.0
                start_idx = int((start_beat+start_loc)*ppbeat)
                # set values to the pianoroll matrix and the onset array
                if settings['velocity'] == 'bool':
                    piano_roll[start_idx:start_idx+1, note.pitch] = True
                elif settings['velocity'] == 'int':
                    piano_roll[start_idx:start_idx+1, note.pitch] = note.velocity
                if start_idx < onset_array.shape[0]:
                    onset_array[start_idx, 0] = True
    return piano_roll, onset_array

# def get_piano_roll_flat(pm, instruments):
#     # Given a list of instruments, return the flatten piano roll
#     piano_roll, onset_array = my_get_piano_roll(pm, instruments[0])
#     for idx in range(len(instruments)-1):
#         next_piano_roll, next_onset_array = my_get_piano_roll(pm, instruments[idx])
#         piano_roll = np.logical_or(piano_roll, next_piano_roll)
#         onset_array = np.logical_or(onset_array, next_onset_array)
#     return piano_roll, onset_array

# def save_piano_roll(piano_roll, dir_path, instrument, family_count=1):
#     """Save piano-roll to a file named <family>_<count> in specific file type ."""
#     try:
#         instrument.family
#     except AttributeError:
#         pretty_midi.program_to_instrument_class(instrument.program)

#     filename = family + '_' + str(family_count) + '.' + settings['filetype']
#     save_ndarray(piano_roll, dir_path, filename)

def save_ndarray(ndarray, dir_path, filename):
    """Save a numpy ndarray to the given directory."""
    filepath = os.path.join(dir_path, filename + settings['filetype'])
    if settings['filetype'] == 'npy':
        np.save(filepath, ndarray)
    elif settings['filetype'] == 'npz':
        sparse_ndarray = scipy.sparse.csc_matrix(ndarray)
        scipy.sparse.save_npz(filepath, sparse_ndarray)
    elif settings['filetype'] == 'csv':
        np.savetxt(filepath, ndarray, delimiter=',')
    else:
        raise TypeError("unknown file type, allowed values are 'npy', 'npz' and 'csv'")

def save_ndarray_dict(ndarray_dict, dir_path):
    """Save a dict of numpy ndarrays to the given directory."""
    for key, value in ndarray_dict.iteritems():
        save_ndarray(value, dir_path, filename=key)

def append_instrument_info_to_dict(instrument_dict, instrument_name, instrument_info):
    instrument_dict[instrument_name] = instrument_info

def extract_instruments(pm, msd_id, midi_md5, beats, tempo_array):
    """Save the piano roll under a sub-folder named by midi_md5
        with instrument category being the file name"""
    song_dir = os.path.join(settings['result_path'], msd_id_to_dirs(msd_id))
    midi_dir = os.path.join(song_dir, midi_md5)
    if not os.path.exists(midi_dir):
        os.makedirs(midi_dir)

    # sort instruments by their program numbers
    pm.instruments.sort(key=lambda x: x.program)
    # iterate thorugh all instruments
    for instrument, idx in enumerate(pm.instruments):
        instrument_info = get_instrument_info(pm, instrument, idx)
        instrument_file_name = '_'.join(str(idx), instrument_info['family name'], instrument_info['program name'])
        # get the piano_roll of a specific instrument
        piano_roll, onset_array = get_piano_roll(pm, instrument, beats=, tempo_array=None)
        # save the piano_roll to file named '<family>_<count_family>', e.g. 'piano_1'
        save_ndarray(piano_roll, midi_dir, instrument_file_name)
        append_instrument_info_to_dict(instrument_dict, file_name, instrument_info)
        count_family[instrument.family] += 1

    # save the instrument dictionary into a json file
    with open(os.path.join(song_dir, 'instruments.json'), 'w') as outfile:
        json.dump(instrument_dict, outfile)
    # copy the selected midi file to the result path
    shutil.copy2(get_midi_path(msd_id, midi_md5), os.path.dirname(midi_dir))






# def generate_midi_dict_entry(pm, max_score):
#     new_entry = {
#         'confidence': max_score,
#         'beat start time': pm.beat_start_time,
#         'incomplete at start': pm.incomplete_at_start,
#         'total beats': pm.num_beats,
#         'total bars': pm.total_bars,
#         'all 2/4': pm.all_two_four,
#         'all 3/4': pm.all_three_four,
#         'all 4/4': pm.all_four_four,
#         'no tsc': pm.no_tsc,
#         'one tsc': pm.one_tsc,
#         'more than one tsc': pm.more_than_one_tsc,
#     }
#     return new_entry

# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO

def get_instrument_statistics(pm, piano_roll, onset_array, tsc_info, beats_info):
    # occurrence ratio
    rhythm_sum = piano_roll.sum(axis=1)
    # if len(rhythm_sum)%96 > 0:
    #     rhythm_sum = np.concatenate((rhythm_sum, np.zeros((96-len(rhythm_sum)%96, ))))
    if tsc_info['num_tsc'] == 1 and tsc_info['time_signature'].endswith('4'):
        bar_sum = np.sum(rhythm_sum.reshape(settings['beat_resolution']*int(tsc_info['time_signature'][0]), -1), axis=0)
        occ_bars = np.sum(bar_sum > 0)
        occurrence_bar_ratio = occ_bars / float(bar_sum.shape[1])
    else:
        occurrence_bar_ratio = None
    # average notes simultaneously
    avg_notes_simultaneously = rhythm_sum.sum() / float((rhythm_sum > 0).sum()) if (rhythm_sum > 0).sum() > 0 else 0.0
    # max notes simultaneously
    max_notes_simultaneously = max(rhythm_sum)
    # rhythm complexity
    rhythm_complexity = float(np.sum(onset_array)) / float(occ_bars) if occ_bars > 0 else 0.0
    # pitch complexity
    actual_bars = int(np.floor(pm.num_beats/4.0))
    pitch_sum = np.sum((np.sum(piano_roll[0:actual_bars*96, :].reshape(-1, 96, 128), axis=1) > 0), axis=1)
    pitch_complexity = np.sum(pitch_sum) / float(occ_bars) if occ_bars > 0 else 0.0

    return {'occurrence bar ratio': occurrence_bar_ratio,
            'average notes simultaneously': avg_notes_simultaneously,
            'max notes simultaneously': max_notes_simultaneously,
            'rhythm complexity': rhythm_complexity,
            'pitch complexity': pitch_complexity}

def get_instrument_info(pm, instrument, identifier):
    instrument_info = {'id': identifier,
                       'program number': instrument.program,
                       'program name': pretty_midi.program_to_instrument_name(instrument.program),
                       'name': instrument.name,
                       'is drum': instrument.is_drum,
                       'family number': int(pretty_midi.program)//8,
                       'family name': pretty_midi.program_to_instrument_class(instrument.program)}
    return instrument_info


if __name__ == "__main__":
    # load score file
    with open(settings['score_file_path']) as f:
        scores = json.load(f)
    # create a dictionary of the selected midi files
    midi_dict = {}
    # traverse from dataset root directory
    for root, dirs, files in os.walk(settings['dataset_path']):
        # look for midi files
        is_midi_dir = False
        for file in files:
            if file.endswith('.mid'):
                msd_id = os.path.basename(os.path.normpath(root))
                is_midi_dir = True
                break

        if is_midi_dir == True:
            # find out the one with the highest matching confidence score
            max_score = 0.0
            for file in files:
                if file.endswith('.mid'):
                    candidate_midi_md5 = os.path.splitext(file)[0]
                    score = scores[msd_id][candidate_midi_md5]
                    if  score >= settings['confidence_threshold'] and score > max_score:
                        max_midi_md5 = os.path.splitext(file)[0]
                        max_score = score

            midi2pianoroll(get_midi_path(msd_id, max_midi_md5))

            # load the MIDI file as a pretty_midi object
            midi_path = get_midi_path(msd_id, max_midi_md5)
            try:
                 pm = pretty_midi.PrettyMIDI(midi_path)
            except:
                continue

            # get time sigature change info and add it to the midi_dict
            midi_dict[msd_id][max_midi_md5]['tsc info'] = get_time_signature_changes_info(pm)

            # get beats info and beats arrays
            beats_info, beats_arrays = get_beats_info_and_arrays(pm)
            midi_dict[msd_id][max_midi_md5]['beats info'] = beats_info
            save_ndarray_dict(beats_arrays, song_dir)

            # get tempo array
            get_tempo_array(pm, beats=beats_arrays['beats'])



            # create and append new entry to midi list
            dict_entry = {'confidence': 0.0,
                          'beat start time': 0.0,
                          'incomplete at start': False,
                          'total bars': 0,
                          'total beats': 0}
            new_entry = generate_midi_dict_entry(pm, max_score)
            midi_dict[msd_id] = {}
            midi_dict[msd_id][max_midi_md5] = new_entry

            # create a dictionary of instruments info
            instrument_dict = {}
            # save beats_array, downbeats_array, tempo_array to files

            save_ndarray(pm.downbeats_array, song_dir, 'downbeats')
            save_ndarray(pm.tempo_array, song_dir, 'tempo')


            # extract all the instrument tracks into piano rolls
            extract_instruments(pm, msd_id, max_midi_md5)



    # print statistics of the elected midi files
    print(len(midi_dict), 'midi files collected')
    for stat in ['no tsc', 'one tsc', 'more than one tsc', 'all 4/4', 'all 3/4', 'all 2/4', 'incomplete at start']:
        print(stat+':', sum((midi_dict[msd_id][midi_md5][stat] == True) for msd_id in midi_dict \
                for midi_md5 in midi_dict[msd_id]))
    print('beat_start_time = 0.0:', sum((midi_dict[msd_id][midi_md5]['beat start time'] == 0.0) \
            for msd_id in midi_dict for midi_md5 in midi_dict[msd_id]))
    # save the midi dict into a json file
    with open(os.path.join(settings['result_path'], 'midi_dict.json'), 'w') as outfile:
        json.dump(midi_dict, outfile)
