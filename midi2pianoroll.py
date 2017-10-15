# Herman Dong 2017.5.18 v1
# Last modified: Herman Dong 2017.09.27 v2
"""
[]

[How to use]
    (1) set the correct DATASET_ROOT, SEARCH_ROOT, RESULTS_PATH constant
    (2) set the correct SCORE_FILE path constant
    (3) set the desired FILETYPE constant
    (4) set the correct CONFIDENCE_TH, BEAT_RESOLUTION constant
    (5) run the following
                python2 run3.py
[Notes]
    Only the midi file with highest confidence score is chosen.
    occurrence ratio: the ratio of the number of bars that the instrument shows up to the number of bars in total
    average notes simultaneously: the average number of notes occurs simultaneously
    max notes simultaneously: the maximum number of notes occurs simultaneously
    rhythm complexity: average number of notes(onset only) in a bar
    pitch complexity: average number of notes with different pitches
"""

# Imports
import numpy as np
import pretty_midi
import os
import shutil
import errno
import json
import scipy.sparse

# Local path constants
DATASET_ROOT = '/home/salu133445/NAS/RichardYang/Dataset/Lakh_MIDI_Dataset'
SEARCH_ROOT = '/home/salu133445/NAS/RichardYang/Dataset/Lakh_MIDI_Dataset/lmd_matched'
RESULTS_PATH = '/home/salu133445/NAS/salu133445/midinet/lmd_parsed'
# Path to the file match_scores.json distributed with the LMD
SCORE_FILE = '/home/salu133445/NAS/RichardYang/Dataset/Lakh_MIDI_Dataset/match_scores.json'

# Parameters
CONFIDENCE_TH = 0.3
BEAT_RESOLUTION = 24
FILETYPE = 'npz'  # 'npy', 'csv'
KIND = 'matched'  # 'aligned'

def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

def get_midi_path(msd_id, midi_md5):
    """Given an MSD ID and MIDI MD5, return path to a MIDI file.
    kind should be one of 'matched' or 'aligned'. """
    return os.path.join(DATASET_ROOT, 'lmd_{}'.format(KIND), msd_id_to_dirs(msd_id), midi_md5 + '.mid')

def instrument_to_category(instrument):
    """Given an instrument object, return its category(string)."""
    if instrument.is_drum :
        return 'Drum'
    elif instrument.program+1  < 9:
        return 'Piano'
    elif instrument.program+1 > 24 and instrument.program+1 < 33:
        return 'Guitar'
    elif instrument.program+1 > 32 and instrument.program+1 < 41:
        return 'Bass'
    elif instrument.program+1 > 40 and instrument.program+1 < 49:
        return 'String'
    else:
        return 'Other'

def get_time_signature_changes_info(pm):
    tsc = pm.time_signature_changes
    # set False as default value of attributes
    setattr(pm, 'no_tsc', False)
    setattr(pm, 'one_tsc', False)
    setattr(pm, 'more_than_one_tsc', False)
    setattr(pm, 'all_four_four', False)
    setattr(pm, 'all_two_four', False)
    setattr(pm, 'all_three_four', False)
    # set attributes True according to tsc properties
    if len(tsc) == 0:
        pm.no_tsc = True;
    elif len(tsc) == 1:
        pm.one_tsc = True;
        if tsc[0].denominator == 4:
            if tsc[0].numerator == 4:
                pm.all_four_four = True;
            elif tsc[0].numerator == 2:
                pm.all_two_four = True;
            elif tsc[0].numerator == 3:
                pm.all_three_four = True;
    else: # len(tsc) > 1
        pm.more_than_one_tsc = True;

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def generate_beats_array(pm):
    # start form time 0.0 or after the first signature changes
    if not pm.time_signature_changes:
        beat_start_time = 0.0
    else:
        beat_start_time = pm.time_signature_changes[0].time
    # use build-in method of pm object to get beats 
    beats = pm.get_beats(beat_start_time)
    total_beats = len(beats)
    # use build-in method of pm object to get down beats 
    downbeats = pm.get_downbeats(beat_start_time)
    # deal with incomplete bar at the very start
    incomplete_at_start = bool(downbeats[0] >  beats[0])
    total_bars = len(downbeats) + int(incomplete_at_start)
    # allocate an empty arrays to fill in
    beats_array = np.zeros(shape=(BEAT_RESOLUTION*total_beats, 1), dtype=bool)
    downbeats_array = np.zeros(shape=(BEAT_RESOLUTION*total_beats, 1), dtype=bool)
    # fill in the beats array and downbeats_array
    beats_array[0:-1:BEAT_RESOLUTION] = True
    for idx in range(len(downbeats)):
        to_fill_idx = np.searchsorted(beats, downbeats[idx], side='right')
        downbeats_array[to_fill_idx] = True
    # save variables to the pm object for future usage
    setattr(pm, 'beat_start_time', beat_start_time)
    setattr(pm, 'beats', beats)
    setattr(pm, 'total_beats', total_beats)
    setattr(pm, 'total_bars', total_bars)
    setattr(pm, 'beats_array', beats_array)
    setattr(pm, 'downbeats_array', downbeats_array)
    setattr(pm, 'incomplete_at_start', incomplete_at_start)

def generate_tempo_array(pm):
    tempo_changes = pm.get_tempo_changes()
    # allocate an empty arrays to fill in
    tempo_array = np.zeros(shape=(BEAT_RESOLUTION*pm.total_beats, 1))
    # use default tempo value when no tempo change events 
    if not tempo_changes:
        tempo_array[:] = 120.0
    else:
        if tempo_changes[0][0] > 0.0:
            tempo_end_beat = (np.searchsorted(pm.beats, tempo_changes[0][0], side='right'))
            tempo_array[0:tempo_change_beat] = tempo_changes[1][0]
            tempo_start_beat = tempo_end_beat
        else:
            tempo_start_beat = 0
        tempo_id = 0
        while tempo_id+1 < len(tempo_changes[0]):
            tempo_end_beat = (np.searchsorted(pm.beats, tempo_changes[0][tempo_id+1], side='right'))
            tempo_array[tempo_start_beat:tempo_end_beat] = tempo_changes[1][tempo_id]
            tempo_start_beat = tempo_end_beat
            tempo_id += 1
        tempo_array[tempo_start_beat:] = tempo_changes[1][tempo_id]
    setattr(pm, 'tempo_array', tempo_array)

def my_get_piano_roll(pm, instrument):
    # allocate an empty matrix to fill in piano roll and an empty onset array
    piano_roll = np.zeros(shape=(BEAT_RESOLUTION*pm.total_beats, 128), dtype=bool)
    onset_array = np.zeros(shape=(BEAT_RESOLUTION*pm.total_beats, 1), dtype=bool)
    # calculate pixel per beat
    ppbeat = BEAT_RESOLUTION
    hppbeat = BEAT_RESOLUTION/2
    min_note_len = 2/BEAT_RESOLUTION
    # iterate through all the note in the instrument track
    if not instrument.is_drum:
        for note in instrument.notes:
            if note.end >= pm.beats[0]:
                if note.start >= pm.beats[0]:
                    start_beat = np.searchsorted(pm.beats, note.start, side='right') - 1
                    start_loc = (note.start - pm.beats[start_beat])
                    start_loc = start_loc * pm.tempo_array[int(start_beat*hppbeat)] / 60.0
                else:
                    start_beat = 0
                    start_loc = 0.0
                end_beat = np.searchsorted(pm.beats, note.end, side='right') - 1
                end_loc = (note.end - pm.beats[end_beat])
                end_loc = end_loc * pm.tempo_array[int(end_beat*hppbeat)] / 60.0
                # calculate the corresponding index in the array
                start_idx = int(start_beat*ppbeat + start_loc*hppbeat)
                end_idx = int(end_beat*ppbeat + end_loc*hppbeat)
                # make sure the note length is larger than minimum note length
                if end_idx - start_idx < 2:
                    end_idx = start_idx + 2 
                piano_roll[start_idx:(end_idx-1), note.pitch] = 1
                if start_idx < onset_array.shape[0]:
                    onset_array[start_idx, 0] = True
            else:
                continue
        return piano_roll, onset_array
    else:  # for drums, we only capture the note on event
        for note in instrument.notes:
            if note.end >= pm.beats[0]:
                if note.start >= pm.beats[0]:
                    start_beat = np.searchsorted(pm.beats, note.start, side='right') - 1
                    start_loc = (note.start - pm.beats[start_beat])
                    start_loc = start_loc * pm.tempo_array[int(start_beat*ppbeat)] / 60.0
                else:
                    start_beat = 0
                    start_loc = 0.0
                start_idx = int((start_beat+start_loc)*ppbeat)
                piano_roll[start_idx:start_idx+1, note.pitch] = 1
                if start_idx < onset_array.shape[0]:
                    onset_array[start_idx, 0] = True
            else:
                continue
        return piano_roll, onset_array

def my_get_piano_roll_flat(pm, instruments):
    # Given a list of instruments, return the flatten piano roll
    piano_roll, onset_array = my_get_piano_roll(pm, instruments[0])
    for idx in range(len(instruments)-1):
        next_piano_roll, next_onset_array = my_get_piano_roll(pm, instruments[idx])
        piano_roll = np.logical_or(piano_roll, next_piano_roll)
        onset_array = np.logical_or(onset_array, next_onset_array)
    return piano_roll, onset_array

def save_piano_roll(piano_roll, category, folder_root, instrument_count=1):
    if instrument_count > 0:
        filepath = os.path.join(folder_root, category + '_' + str(instrument_count) + '.' + FILETYPE)
    else:
        filepath = os.path.join(folder_root, category + '.' + FILETYPE)
    if FILETYPE == 'npy':  # uncompressed numpy matrix
        np.save(filepath, piano_roll)
    elif FILETYPE == 'csv':
        np.savetxt(filepath, piano_roll, delimiter=',')
    elif FILETYPE == 'npz':  # compressed scipy sparse matrix
        sparse_piano_roll = scipy.sparse.csc_matrix(piano_roll)
        scipy.sparse.save_npz(filepath, sparse_piano_roll)
    else:
        print 'Error: Unknown file type'

def generate_instrument_dict_entry(pm, instrument, piano_roll, onset_array, instrument_count=1):
    # calculate instrument statistics
    rhythm_sum = piano_roll.sum(axis=1)
    if len(rhythm_sum)%96 > 0:
        rhythm_sum = np.concatenate((rhythm_sum, np.zeros((96-len(rhythm_sum)%96, ))))
    bar_sum = np.sum(rhythm_sum.reshape(96, -1), axis=0)
    occ_bars = np.sum(bar_sum>0)
    occurrence_ratio = occ_bars / float(pm.total_bars)
    if (rhythm_sum>0).sum() > 0:
        avg_notes_simultaneously = rhythm_sum.sum() / float((rhythm_sum>0).sum())
    else:
        avg_notes_simultaneously = 0.0
    max_notes_simultaneously = max(rhythm_sum)
    actual_bars = int(np.floor(pm.total_beats/4.0))
    pitch_sum = np.sum(piano_roll[0:actual_bars*96, :].reshape(-1, 96,128), axis=1)
    pitch_sum = np.sum((pitch_sum>0), axis=1)
    if occ_bars > 0:
        rhythm_complexity = float(np.sum(onset_array)) / float(occ_bars)
        pitch_complexity = np.sum(pitch_sum) / float(occ_bars)
    else:
        rhythm_complexity = 0.0
        pitch_complexity = 0.0
    # create new entry according to instrument types
    new_entry = {
        'occurrence ratio': occurrence_ratio,
        'average notes simultaneously': avg_notes_simultaneously,
        'max notes simultaneously': max_notes_simultaneously,
        'rhythm complexity': rhythm_complexity,
        'pitch complexity': pitch_complexity,
        'instrument count': instrument_count
    }
    if instrument is not 'Drum' and instrument is not 'Bass':
        new_entry['program number'] = instrument.program
        new_entry['instrument name'] = instrument.name
        new_entry['instrument category'] = instrument_to_category(instrument)
        new_entry['is drum'] = False
    else:
        new_entry['program number'] = -1
        new_entry['instrument category'] = instrument
        new_entry['instrument name'] = instrument
        if instrument == 'Drum':
            new_entry['is drum'] = True
        else: # 'Bass'
            new_entry['is drum'] = False
    return new_entry

def extract_instruments(pm, msd_id, midi_md5):
    """Save the piano roll under a sub-folder named by midi_md5
        with instrument category being the file name"""
    midi_folder_root = os.path.join(RESULTS_PATH, 'lmd_{}'.format(KIND), msd_id_to_dirs(msd_id), midi_md5)
    make_sure_path_exists(midi_folder_root)
    # create a dictionary of instruments info
    instrument_dict = {}
    count_instrument = {'Piano': 1, 'Guitar': 1, 'String': 1, 'Other': 1, 'Bass': 1, 'Drum': 1}
    # assign instruments into groups
    drums = [x for x in pm.instruments if x.is_drum == True]
    bass = [x for x in pm.instruments if instrument_to_category(x) == 'Bass']
    others = [x for x in pm.instruments if instrument_to_category(x) is not 'Bass' and x.is_drum == False]
    others.sort(key=lambda x: x.program)
    # deal with drums and bass
    for x in ['Drum', 'Bass']:
        if x == 'Drum':
            if not drums:
                continue
            piano_roll, onset_array = my_get_piano_roll_flat(pm, drums)
        else:
            if not bass:
                continue
            piano_roll, onset_array = my_get_piano_roll_flat(pm, bass)
        save_piano_roll(piano_roll, x, midi_folder_root)
        new_entry = generate_instrument_dict_entry(pm, x, piano_roll, onset_array)
        instrument_dict[x+ '_1'] = new_entry
    # deal with other instruments
    for i in range(len(others)):
        piano_roll, onset_array = my_get_piano_roll(pm, others[i])
        category = instrument_to_category(others[i])
        save_piano_roll(piano_roll, category, midi_folder_root, count_instrument[category])
        new_entry = generate_instrument_dict_entry(pm, others[i], piano_roll, onset_array, count_instrument[category])
        instrument_dict[category + '_' + str(count_instrument[category])] = new_entry
        count_instrument[category] += 1
    # save downbeats_array, tempo_array
    save_piano_roll(pm.downbeats_array, 'Downbeats', os.path.dirname(midi_folder_root), 0)
    save_piano_roll(pm.tempo_array, 'Tempo', os.path.dirname(midi_folder_root), 0)
    # save the instrument dictionary into a json file
    with open(os.path.join(os.path.dirname(midi_folder_root), 'instruments.json'), 'w') as outfile:
        json.dump(instrument_dict, outfile)
    # copy the selected midi file to the result path
    shutil.copy2(get_midi_path(msd_id, midi_md5), os.path.dirname(midi_folder_root))

def generate_midi_dict_entry(pm, max_score):
    new_entry = {
        'confidence': max_score,
        'beat start time': pm.beat_start_time,
        'incomplete at start': pm.incomplete_at_start,
        'total beats': pm.total_beats,
        'total bars': pm.total_bars,
        'all 2/4': pm.all_two_four,
        'all 3/4': pm.all_three_four,
        'all 4/4': pm.all_four_four,
        'no tsc': pm.no_tsc,
        'one tsc': pm.one_tsc,
        'more than one tsc': pm.more_than_one_tsc,
        'total bars': pm.total_bars,
        'total beats': pm.total_beats
    }
    return new_entry

if __name__ == "__main__":
    with open(SCORE_FILE) as f:
        scores = json.load(f)
    # create a dictionary of the selected midi files
    midi_dict = {}
    # traverse from SEARCH_ROOT
    for root, dirs, files in os.walk(SEARCH_ROOT):
        msd_id = os.path.basename(os.path.normpath(root))
        max_score = 0.0
        any_midi_found = False
        for file in files:
            if file.endswith(".mid"):
                any_midi_found = True
                midi_md5 = os.path.splitext(file)[0]
                score = scores[msd_id][midi_md5]
                # Find out the midi file with the highest confidence score
                if  score > CONFIDENCE_TH and score > max_score:
                    max_midi_md5 = os.path.splitext(file)[0]
                    max_score = score
        if any_midi_found == True:
            # Construct the path to the MIDI
            midi_path = get_midi_path(msd_id, midi_md5)
            # Load/parse the MIDI file with pretty_midi
            try:  
                 pm = pretty_midi.PrettyMIDI(midi_path)
            except:  
                continue
            # check the number of time signature changes
            get_time_signature_changes_info(pm)
            # generate beats and tempo array
            generate_beats_array(pm)
            generate_tempo_array(pm)
            # create and append new entry to midi list
            new_entry = generate_midi_dict_entry(pm, max_score)
            midi_dict[msd_id] = {}
            midi_dict[msd_id][max_midi_md5] = new_entry
            # extract all the instrument tracks into piano rolls
            extract_instruments(pm, msd_id, max_midi_md5)
    # print statistics of the elected midi files
    print len(midi_dict), 'midi files collected'
    for stat in ['no tsc', 'one tsc', 'more than one tsc', 'all 4/4', 'all 3/4', 'all 2/4', 'incomplete at start']:
        print stat+':', sum((midi_dict[msd_id][midi_md5][stat] == True) for msd_id in midi_dict \
                for midi_md5 in midi_dict[msd_id])
    print 'beat_start_time = 0.0:', sum((midi_dict[msd_id][midi_md5]['beat start time'] == 0.0) \
            for msd_id in midi_dict for midi_md5 in midi_dict[msd_id])
    # save the midi dict into a json file
    with open(os.path.join(RESULTS_PATH, 'midi_dict.json'), 'w') as outfile:
        json.dump(midi_dict, outfile)
