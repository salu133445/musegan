import numpy as np
import os
from pypianoroll import Multitrack, Track
import json
import pickle
import argparse
from pathlib import Path
from glob import glob

family_name = [
    "drum",
    "bass",
    "guitar",
    "string",
    "piano",
]

family_thres = [
    (2, 24),  # drum
    (1, 96),  # bass
    (2, 156),  # guitar
    (2, 156),  # string,
    (2, 156),  # piano
]


def parse():
    parser = argparse.ArgumentParser(
        description="Convert midi files into training set")
    parser.add_argument("dir",
                        help="directory containing .mid files")
    parser.add_argument("-r", "--recursive", dest="recursive", action="store_true",
                        help="search directory recursively")
    parser.add_argument("--outfile", dest="out",
                        default="data/train.npy",
                        help="output file")

    return parser.parse_args()

def check_which_family(track):
    def is_piano(program, is_drum): return not is_drum and ((program >= 0 and program <= 7)
                                                            or (program >= 16 and program <= 23))

    def is_guitar(program): return program >= 24 and program <= 31

    def is_bass(program): return program >= 32 and program <= 39

    def is_string(program): return program >= 40 and program <= 51

    # drum, bass, guitar, string, piano
    def is_instr_act(program, is_drum): return np.array([is_drum, is_bass(program), is_guitar(program),
                                                         is_string(program), is_piano(program, is_drum)])

    instr_act = is_instr_act(track.program, track.is_drum)
    return instr_act

def segment_quality(pianoroll, thres_pitch, thres_beats):
    pitch_sum = sum(np.sum(pianoroll.pianoroll, axis=0) > 0)
    beat_sum = sum(np.sum(pianoroll.pianoroll, axis=1) > 0)
    score = pitch_sum + beat_sum
    return (pitch_sum >= thres_pitch) and (beat_sum >= thres_beats), (pitch_sum, beat_sum)


if __name__ == "__main__":

    num_consecutive_bar = 4
    resol = 12
    down_sample = 2
    cnt_total_segments = 0
    cnt_augmented = 0
    ok_segment_list = []
    hop_size = (num_consecutive_bar / 4)
    args = parse()
    recursive = args.recursive
    dir = args.dir
    if not os.path.isdir(dir):
        raise argparse.ArgumentTypeError("dir must be a directory")
    outfile = args.out
    if not outfile.endswith(".npy") and not outfile.endswith(".npz"):
        outfile += ".npy"
    try:
        outdir = os.path.split(outfile)[0]
        if outdir != "" and not os.path.exists(outdir):
            os.makedirs(outdir)
        open(outfile, "w").close()
    except Exception as e:
        raise argparse.ArgumentTypeError("outfile is not valid")
    for file in (Path(dir).rglob("*.mid") if recursive else glob(f"{dir}/*.mid")):
        print(f"Processing {file}")
        multitrack = Multitrack(file)
        downbeat = multitrack.downbeat

        num_bar = len(downbeat) // resol
        hop_iter = 0

        song_ok_segments = []
        for bidx in range(num_bar-num_consecutive_bar):
            if hop_iter > 0:
                hop_iter -= 1
                continue

            st = bidx * resol
            ed = st + num_consecutive_bar * resol

            best_instr = [Track(pianoroll=np.zeros(
                (num_consecutive_bar*resol, 128)))] * 5
            best_score = [-1] * 5
            second_act = [False] * 5
            second_instr = [None] * 5
            for tidx, track in enumerate(multitrack.tracks):
                tmp_map = check_which_family(track)
                in_family = np.where(tmp_map)[0]

                if not len(in_family):
                    continue
                family = in_family[0]

                tmp_pianoroll = track[st:ed:down_sample]
                is_ok, score = segment_quality(
                    tmp_pianoroll, family_thres[family][0], family_thres[family][1])

                if is_ok and sum(score) > best_score[family]:
                    track.name = family_name[family]
                    best_instr[family] = track[st:ed:down_sample]
                    best_score[family] = sum(score)

            hop_iter = np.random.randint(0, 1) + hop_size
            song_ok_segments.append(Multitrack(
                tracks=best_instr, beat_resolution=12))

        cnt_ok_segment = len(song_ok_segments)
        if cnt_ok_segment > 6:
            seed = (6, cnt_ok_segment//2)
            if cnt_ok_segment > 11:
                seed = (11, cnt_ok_segment//3)
            if cnt_ok_segment > 15:
                seed = (15, cnt_ok_segment//4)

            rand_idx = np.random.permutation(cnt_ok_segment)[:max(seed)]
            song_ok_segments = [song_ok_segments[ridx] for ridx in rand_idx]
            ok_segment_list.extend(song_ok_segments)
            cnt_ok_segment = len(rand_idx)
        else:
            ok_segment_list.extend(song_ok_segments)

        cnt_total_segments += len(song_ok_segments)
        print(f"current: {cnt_ok_segment} | cumulative: {cnt_total_segments}")
    print("-"*30)
    print(cnt_total_segments)
    num_item = len(ok_segment_list)
    compiled_list = []
    for lidx in range(num_item):
        multi_track = ok_segment_list[lidx]
        pianorolls = []

        for tracks in multi_track.tracks:
            pianorolls.append(tracks.pianoroll[:, :, np.newaxis])

        pianoroll_compiled = np.reshape(np.concatenate(pianorolls, axis=2)[
                                        :, 24:108, :], (num_consecutive_bar, resol, 84, 5))
        pianoroll_compiled = pianoroll_compiled[np.newaxis, :] > 0
        compiled_list.append(pianoroll_compiled.astype(bool))
    result = np.concatenate(compiled_list, axis=0)
    print(f"output shape: {result.shape}")
    if outfile.endswith(".npz"):
        np.savez_compressed(
            outfile, nonzero=np.array(result.nonzero()),
            shape=result.shape)
    else:
        np.save(outfile, result)
    print(f"saved to {outfile}")