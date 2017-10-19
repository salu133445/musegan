"""Utility functions for conversion between midi and pianoroll"""

import numpy as np
import pretty_midi


def merge_dicts(*dict_args):
    """Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts."""
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def get_time_signature_info_and_arrays(pm):
    """Given a pretty_midi.PrettyMIDI class instance, return its time signature
    info dictionary and time signature array dictionary."""
    # sort time signature changes by time
    pm.time_signature_changes.sort(key=lambda x: x.time)
    # create a list of time signature changes content and another for the event times
    time_signature_times = [tsc.time for tsc in pm.time_signature_changes]
    time_signature_numerators = [tsc.numerator for tsc in pm.time_signature_changes]
    time_signature_denominators = [tsc.denominator for tsc in pm.time_signature_changes]
    # set time signature if only one time signature change event found
    if len(time_signature_times) == 1:
        time_signature = str(time_signature_numerators[0]) + '/' + str(time_signature_denominators[0])
    else:
        time_signature = None
    # collect variables into dictionaries to return
    time_signature_info = {'num_time_signature_changes': len(time_signature_times),
                           'time_signature': time_signature}
    time_signature_arrays = {'time_signature_numerators': np.array(time_signature_numerators),
                             'time_signature_denominators': np.array(time_signature_denominators),
                             'time_signature_times': np.array(time_signature_times)}
    return time_signature_info, time_signature_arrays

def get_beat_info_and_arrays(pm, beat_resolution=24, sort_tsc=True):
    """Given a pretty_midi.PrettyMIDI class instance, return its beat info
    dictionary and beat array dictionary. If sort_tsc is True(by default), the
    time_signatrue_changes list of the pretty_midi object will first be sorted.
    """
    # sort time signature changes by time
    if sort_tsc:
        pm.time_signature_changes.sort(key=lambda x: x.time)
    # use built-in method in pretty_midi to get beat_start_time, beats and downbeats
    beat_start_time = pm.time_signature_changes[0].time if pm.time_signature_changes else 0.0
    beat_times = pm.get_beats(beat_start_time)
    downbeat_times = pm.get_downbeats(beat_start_time)
    # calculate beats information
    num_beats = len(beat_times)
    incomplete_at_start = bool(downbeat_times[0] > beat_times[0])
    num_bars = len(downbeat_times) + int(incomplete_at_start)
    # create an empty beat array and an empty downbeat array
    beat_array = np.zeros(shape=(beat_resolution*num_beats, 1), dtype=bool)
    downbeat_array = np.zeros(shape=(beat_resolution*num_beats, 1), dtype=bool)
    # fill in the beats array and the downbeats array
    beat_array[0:-1:beat_resolution] = True
    for _, downbeat_time in enumerate(downbeat_times):
        idx_to_fill = np.searchsorted(beat_times, downbeat_time, side='right')
        downbeat_array[idx_to_fill] = True
    # collect variables into dictionaries to return
    beat_info = {'beat_start_time': beat_start_time,
                 'num_beats': num_beats,
                 'num_bars': num_bars,
                 'incomplete_at_start': incomplete_at_start}
    beat_arrays = {'beat_times': beat_times,
                   'downbeat_times': downbeat_times,
                   'beat_array': beat_array,
                   'downbeat_array': downbeat_array}
    return beat_info, beat_arrays

def get_tempo_info_and_arrays(pm, beat_resolution=24, beat_times=None):
    """Given a pretty_midi.PrettyMIDI class instance, return its tempo info
    dictionary and tempo info array dictionary. If no beat_times is given,
    pm.get_beats(beat_start_time) will be first computed to get beat_times."""
    # compute beat_times when it is not given
    if beat_times is None:
        beat_start_time = pm.time_signature_changes[0].time if pm.time_signature_changes else 0.0
        beat_times = pm.get_beats(beat_start_time)
    # create an empty tempo_array
    tempo_array = np.zeros(shape=(beat_resolution*len(beat_times), 1), dtype=float)
    # use built-in method in pretty_midi to get tempo change events
    tempo_change_times, tempi = pm.get_tempo_changes()
    if not tempo_change_times.size:
        # set to default tempo value when no tempo change events
        tempo_array[:] = 120.0
    else:
        # deal with the first tempo change event
        tempo_end_beat = (np.searchsorted(beat_times, tempo_change_times[0], side='right'))
        tempo_array[0:tempo_end_beat] = tempi[0]
        tempo_start_beat = tempo_end_beat
        # deal with the rest tempo change event
        tempo_id = 0
        while tempo_id+1 < len(tempo_change_times):
            tempo_end_beat = (np.searchsorted(beat_times, tempo_change_times[tempo_id+1], side='right'))
            tempo_array[tempo_start_beat:tempo_end_beat] = tempi[tempo_id]
            tempo_start_beat = tempo_end_beat
            tempo_id += 1
        # deal with the rest beats
        tempo_array[tempo_start_beat:] = tempi[tempo_id]
    # collect variables into dictionaries to return
    tempo_info = {'tempo': tempi[0] if len(tempo_change_times) == 1 else None}
    tempo_arrays = {'tempo_change_times': tempo_change_times,
                    'tempi': tempi,
                    'tempo_array': tempo_array}
    return tempo_info, tempo_arrays

def get_midi_info_and_arrays(pm, beat_resolution=24):
    """Given a pretty_midi.PrettyMIDI class instance, return its midi_info_dict
    and midi_array_dict."""
    # get time sigature changes info
    time_signature_info, time_signature_arrays = get_time_signature_info_and_arrays(pm)
    # get beat info dictionary and beats array dictionary
    beat_info, beat_arrays = get_beat_info_and_arrays(pm, beat_resolution=beat_resolution, sort_tsc=False)
    # get tempo info dictionary and tempo array dictionary
    tempo_info, tempo_arrays = get_tempo_info_and_arrays(pm, beat_resolution=beat_resolution,
                                                         beat_times=beat_arrays['beat_times'])
    # collect the results into dictionaries to return
    midi_info = merge_dicts(beat_info, time_signature_info, tempo_info)
    midi_arrays = merge_dicts(beat_arrays, time_signature_arrays, tempo_arrays)
    return midi_info, midi_arrays

def get_piano_roll(instrument, beat_resolution=24, beat_times=None, tempo_array=None, pm=None):
    """Given a pretty_midi.Instrument class instance, return the pianoroll of
    the instrument. When one of the beat_times and the tempo_array is not given,
    the pretty_midi object should be given."""
    if beat_times is None:
        beat_start_time = pm.time_signature_changes[0].time if pm.time_signature_changes else 0.0
        beat_times = pm.get_beats(beat_start_time)
    if tempo_array is None:
        _, tempo_array = get_tempo_info_and_arrays(pm, beat_times)
    num_beats = len(beat_times)
    # create the piano roll and the onset roll
    piano_roll = np.zeros(shape=(beat_resolution*num_beats, 128), dtype=int)
    onset_roll = np.zeros(shape=(beat_resolution*num_beats, 1), dtype=bool)
    # calculate pixel per beat
    ppbeat = beat_resolution
    hppbeat = beat_resolution/2
    # iterate through notes
    for note in instrument.notes:
        if note.end < beat_times[0]:
            continue
        else:
            # find the corresponding index of the note on event
            if note.start >= beat_times[0]:
                start_beat = np.searchsorted(beat_times, note.start, side='right') - 1
                start_loc = (note.start - beat_times[start_beat])
                start_loc = start_loc * tempo_array[int(start_beat*hppbeat)] / 60.0
            else:
                start_beat = 0
                start_loc = 0.0
            start_idx = int(start_beat*ppbeat + start_loc*hppbeat)
            # find the corresponding index of the note off event
            if instrument.is_drum:
                # set note length to minimal (32th notes) for drums
                end_idx = start_idx + 2
            else:
                end_beat = np.searchsorted(beat_times, note.end, side='right') - 1
                end_loc = (note.end - beat_times[end_beat])
                end_loc = end_loc * tempo_array[int(end_beat*hppbeat)] / 60.0
                end_idx = int(end_beat*ppbeat + end_loc*hppbeat)
                # make sure the note length is larger than minimum note length
                if end_idx - start_idx < 2:
                    end_idx = start_idx + 2
            # set values to the piano-roll and the onset-roll matrix
            piano_roll[start_idx:(end_idx-1), note.pitch] = note.velocity
            if start_idx < onset_roll.shape[0]:
                onset_roll[start_idx, 0] = True
    return piano_roll, onset_roll

def get_instrument_info(instrument):
    """Given a pretty_midi.Instrument class instance, return the infomation
    dictionary of the instrument."""
    return {'program_num': instrument.program,
            'program_name': pretty_midi.program_to_instrument_name(instrument.program),
            'name': instrument.name.strip(),
            'is_drum': instrument.is_drum,
            'family_num': int(instrument.program)//8,
            'family_name': pretty_midi.program_to_instrument_class(instrument.program)}

def get_piano_rolls(pm, beat_resolution=24):
    """
    Convert a midi file to piano-rolls of multiple tracks.

    Parameters
    ----------
    midi_path : str
        The path to the midi file.

    Returns
    -------
    piano_rolls : np.ndarray of int
        The extracted piano-rolls. The value represents the velocity. The first
        dimension is the id of the instrument. The size is (num_instrument,
        num_time_step, num_pitches).
    onset_rolls : np.ndarray of bool
        The extracted onset-rolls. The value indicates the occurence of onset
        events. The first dimension is the id of the instrument. The size is
        (num_instrument, num_time_step, num_pitches).
    info_dict : dict
        A dictionary containing extracted useful information lost during the
        conversion of the midi file.
            midi_arrays : dict
                A dictionary containing informative arrays.
                    beat_times : np.ndarray
                        The time (in sec) of each beat
                    downbeat_times : np.ndarray
                        The time (in sec) of each downbeat
                    tempo_array : np.ndarray
                        The tempo at each time step
                    beat_array : np.ndarray
                        The location (time step) of beats
                    downbeat_array : np.ndarray
                        The location (time step) of downbeats
            midi_info : dict
                Contains information of the midi file, including time_signature,
                beat and tempo info.
            instrument_info: dict
                Contains information of each track
    """
    # create an empty instrument dictionary to store information of each instrument
    instrument_info = {}
    piano_rolls = []
    onset_rolls = []
    # get the midi information and the beat/tempo arrays
    midi_info, midi_arrays = get_midi_info_and_arrays(pm, beat_resolution)
    # sort instruments by their program numbers
    pm.instruments.sort(key=lambda x: x.program)
    # iterate thorugh all instruments
    for idx, instrument in enumerate(pm.instruments):
        # get the piano-roll and the onset-roll of a specific instrument
        piano_roll, onset_roll = get_piano_roll(instrument, beat_resolution=beat_resolution,
                                                beat_times=midi_arrays['beat_times'],
                                                tempo_array=midi_arrays['tempo_array'])
        # append the piano-roll to the piano-roll list and the onset-roll list
        piano_rolls.append(piano_roll)
        onset_rolls.append(onset_roll)
        # append information of current instrument to instrument dictionary
        instrument_info[str(idx)] = get_instrument_info(instrument)
    info_dict = {'midi_arrays': midi_arrays,
                 'midi_info': midi_info,
                 'instrument_info': instrument_info}
    return piano_rolls, onset_rolls, info_dict

def midi_to_pianorolls(midi_path, beat_resolution=24):
    """
    Convert a midi file to piano-rolls of multiple tracks.

    Parameters
    ----------
    midi_path : str
        The path to the midi file.

    Returns
    -------
    piano_rolls : np.ndarray of int
        The extracted piano-rolls. The value represents the velocity. The first
        dimension is the id of the instrument. The size is (num_instrument,
        num_time_step, num_pitches).
    onset_rolls : np.ndarray of bool
        The extracted onset-rolls. The value indicates the occurence of onset
        events. The first dimension is the id of the instrument. The size is
        (num_instrument, num_time_step, num_pitches).
    info_dict : dict
        A dictionary containing extracted useful information lost during the
        conversion of the midi file.
            midi_arrays : dict
                A dictionary containing informative arrays.
                    beat_times : np.ndarray
                        The time (in sec) of each beat
                    downbeat_times : np.ndarray
                        The time (in sec) of each downbeat
                    tempo_array : np.ndarray
                        The tempo at each time step
                    beat_array : np.ndarray
                        The location (time step) of beats
                    downbeat_array : np.ndarray
                        The location (time step) of downbeats
            midi_info : dict
                Contains information of the midi file, including time_signature,
                beat and tempo info.
            instrument_info: dict
                Contains information of each track
    """
    # load the MIDI file as a pretty_midi object
    pm = pretty_midi.PrettyMIDI(midi_path)
    return get_piano_rolls(pm, beat_resolution)
