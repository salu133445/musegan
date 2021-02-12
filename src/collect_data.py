"""Collect training data from MIDI files."""
import argparse
from pathlib import Path

import numpy as np
from pypianoroll import Multitrack, Track

FAMILY_NAMES = [
    "drum",
    "bass",
    "guitar",
    "string",
    "piano",
]

FAMILY_THRESHOLDS = [
    (2, 24),  # drum
    (1, 96),  # bass
    (2, 156),  # guitar
    (2, 156),  # string,
    (2, 156),  # piano
]


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect training data from MIDI files"
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=Path,
        required=True,
        help="directory containing MIDI files",
    )
    parser.add_argument(
        "-o",
        "--output_filename",
        type=Path,
        required=True,
        help="output filename",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="whether to search directory recursively",
    )
    return parser.parse_args()


def check_which_family(track):
    def is_piano(program, is_drum):
        return not is_drum and (
            (program >= 0 and program <= 7)
            or (program >= 16 and program <= 23)
        )

    def is_guitar(program):
        return program >= 24 and program <= 31

    def is_bass(program):
        return program >= 32 and program <= 39

    def is_string(program):
        return program >= 40 and program <= 51

    # drum, bass, guitar, string, piano
    def is_instr_act(program, is_drum):
        return np.array(
            [
                is_drum,
                is_bass(program),
                is_guitar(program),
                is_string(program),
                is_piano(program, is_drum),
            ]
        )

    instr_act = is_instr_act(track.program, track.is_drum)
    return instr_act


def segment_quality(pianoroll, threshold_pitch, threshold_beats):
    pitch_sum = np.sum(np.sum(pianoroll.pianoroll, axis=0) > 0)
    beat_sum = np.sum(np.sum(pianoroll.pianoroll, axis=1) > 0)
    return (
        (pitch_sum >= threshold_pitch) and (beat_sum >= threshold_beats),
        (pitch_sum, beat_sum),
    )


def main():
    """Main function."""
    num_consecutive_bar = 4
    resolution = 12
    down_sample = 2
    count_total_segments = 0
    ok_segment_list = []
    hop_size = num_consecutive_bar / 4
    args = parse_arguments()

    if args.recursive:
        filenames = args.input_dir.rglob("*.mid")
    else:
        filenames = args.input_dir.glob("*.mid")

    for filename in filenames:
        print(f"Processing {filename}")
        multitrack = Multitrack(filename)
        downbeat = multitrack.downbeat

        num_bar = len(downbeat) // resolution
        hop_iter = 0

        song_ok_segments = []
        for bidx in range(num_bar - num_consecutive_bar):
            if hop_iter > 0:
                hop_iter -= 1
                continue

            st = bidx * resolution
            ed = st + num_consecutive_bar * resolution

            best_instr = [
                Track(
                    pianoroll=np.zeros((num_consecutive_bar * resolution, 128))
                )
            ] * 5
            best_score = [-1] * 5
            for track in multitrack.tracks:
                tmp_map = check_which_family(track)
                in_family = np.where(tmp_map)[0]

                if not len(in_family):
                    continue
                family = in_family[0]

                tmp_pianoroll = track[st:ed:down_sample]
                is_ok, score = segment_quality(
                    tmp_pianoroll,
                    FAMILY_THRESHOLDS[family][0],
                    FAMILY_THRESHOLDS[family][1],
                )

                if is_ok and sum(score) > best_score[family]:
                    track.name = FAMILY_NAMES[family]
                    best_instr[family] = track[st:ed:down_sample]
                    best_score[family] = sum(score)

            hop_iter = np.random.randint(0, 1) + hop_size
            song_ok_segments.append(
                Multitrack(tracks=best_instr, beat_resolution=12)
            )

        count_ok_segment = len(song_ok_segments)
        if count_ok_segment > 6:
            seed = (6, count_ok_segment // 2)
            if count_ok_segment > 11:
                seed = (11, count_ok_segment // 3)
            if count_ok_segment > 15:
                seed = (15, count_ok_segment // 4)

            rand_idx = np.random.permutation(count_ok_segment)[: max(seed)]
            song_ok_segments = [song_ok_segments[ridx] for ridx in rand_idx]
            ok_segment_list.extend(song_ok_segments)
            count_ok_segment = len(rand_idx)
        else:
            ok_segment_list.extend(song_ok_segments)

        count_total_segments += len(song_ok_segments)
        print(
            f"current: {count_ok_segment} | cumulative: {count_total_segments}"
        )

    print("-" * 30)
    print(count_total_segments)
    num_item = len(ok_segment_list)
    compiled_list = []
    for lidx in range(num_item):
        multi_track = ok_segment_list[lidx]
        pianorolls = []

        for tracks in multi_track.tracks:
            pianorolls.append(tracks.pianoroll[:, :, np.newaxis])

        pianoroll_compiled = np.reshape(
            np.concatenate(pianorolls, axis=2)[:, 24:108, :],
            (num_consecutive_bar, resolution, 84, 5),
        )
        pianoroll_compiled = pianoroll_compiled[np.newaxis, :] > 0
        compiled_list.append(pianoroll_compiled.astype(bool))

    result = np.concatenate(compiled_list, axis=0)
    print(f"output shape: {result.shape}")
    if args.outfile.endswith(".npz"):
        np.savez_compressed(
            args.outfile,
            nonzero=np.array(result.nonzero()),
            shape=result.shape,
        )
    else:
        np.save(args.outfile, result)
    print(f"Successfully saved training data to : {args.outfile}")


if __name__ == "__main__":
    main()
