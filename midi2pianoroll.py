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


def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

def get_midi_path(msd_id, midi_md5):
    """Given an MSD ID and MIDI MD5, return path to a MIDI file.
    kind should be one of 'matched' or 'aligned'. """
    return os.path.join(settings['dataset_path'], msd_id_to_dirs(msd_id), midi_md5 + '.mid')

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

def get_midi_info_and_arrays(pm)
    # get time sigature change info and add it to the midi_dict
    tsc_info = get_time_signature_changes_info(pm)
    # get beats info and beats arrays
    beats_info, beats_arrays = get_beats_info_and_arrays(pm)
    # get tempo array
    tempo_array = get_tempo_array(pm, beats=beats_arrays['beats'])
    # pack info and arrays into dictionaries
    midi_info_dict = {'tsc_info': tsc_info, 'beats_info': beats_info}
    array_dict = {'beats_arrays': beats_arrays, 'tempo_array': tempo_array}
    return midi_info_dict, array_dict

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

def get_instrument_info(pm, instrument, identifier):
    instrument_info = {'id': identifier,
                       'program number': instrument.program,
                       'program name': pretty_midi.program_to_instrument_name(instrument.program),
                       'name': instrument.name,
                       'is drum': instrument.is_drum,
                       'family number': int(pretty_midi.program)//8,
                       'family name': pretty_midi.program_to_instrument_class(instrument.program)}
    return instrument_info

def extract_instruments(pm, array_dict):
    """Save the piano roll under a sub-folder named by midi_md5
        with instrument category being the file name"""
    # create an empty instrument dictionary to store information of each instrument
    instrument_dict = {}
    # sort instruments by their program numbers
    pm.instruments.sort(key=lambda x: x.program)
    # iterate thorugh all instruments
    for instrument, idx in enumerate(pm.instruments):
        # get the piano_roll and the onset array of a specific instrument
        piano_roll, onset_array = get_piano_roll(pm, instrument, beats=beats, tempo_array=tempo_array)
        # append onset array to the array dictionary
        array_dict['onset_array'] = onset_array
        # append information of current instrument to instrument dictionary
        instrument_dict[instrument_name] = get_instrument_info(pm, instrument, idx)
    return piano_roll_dict, instrument_dict

def midi_to_pianoroll(midi_path):
    """
    Convert a midi file to piano-rolls of multiple tracks.
# TODO TODO TODO TODO TODO
    Args:
        midi_path (str): The path to the midi file.
        output_size (int): The output size.
        stddev (float): The value passed to the truncated normal initializer for weights. Defaults to 0.02.
        bias_init (float): The value passed to constant initializer for weights. Defaults to 0.0.
        name (str): The tenorflow variable scope. Defaults to 'linear'.
        reuse (bool): True to reuse weights and biases.
        padding (str): 'SAME' or 'VALID'. The type of padding algorithm to use. Defaults to 'VALID'.

    Returns:
        dict: The resulting tensor.

    """
    # load the MIDI file as a pretty_midi object
    pm = pretty_midi.PrettyMIDI(midi_path)
    # get the midi information and the beats/tempo arrays
    midi_info_dict, array_dict = get_midi_info_and_arrays(pm)
    # extract all the instrument tracks into piano rolls
    piano_roll_dict, instrument_dict = extract_instruments(pm, array_dict)
    return piano_roll_dict, instrument_dict, midi_info_dict, array_dict
