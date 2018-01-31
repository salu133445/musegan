# Dataset

## Lakh Pianoroll Dataset (LPD)

The Lakh Pianoroll Dataset (LPD) is a collection of 174,154 unique [multi-track piano-rolls](#multitrack) derived from the MIDI files in [Lakh MIDI Dataset](http://colinraffel.com/projects/lmd/) (LMD).

- [lpd-full](https://drive.google.com/file/d/1nhYAYIOULo0ti_qnJ_FIfhs0A-2bdLTK/view?usp=drivesdk):
174,154 multi-track piano-rolls derived from midi files in LMD.
- [lpd-matched](https://drive.google.com/file/d/100D07HPFWKPOoODihykiMK5HH0lX7XlQ/view?usp=drivesdk)\*:
115,160 multi-track piano-rolls derived from midi files of 30,886 songs in LMD-matched.
- [lpd-cleansed](https://drive.google.com/file/d/13FZghe-Slw7TUT3YZLzEGP5iV7e1T_NV/view?usp=drivesdk)\*:
24,474 multi-track piano-rolls derived from midi files of distinct songs in LMD-matched.

The following is LPD-5, another version of LPD. In LPD-5, the tracks of each multi-track piano-rolls are merged into five common categories: bass, drums, piano, guitar and strings according to the program numbers provided in the MIDI files.
Note that instruments out of the list are considered as part of the strings

- [lpd-5-full](https://drive.google.com/file/d/1etN6WPDxddApbGw-ZuCuv9txRnhe1Wd1/view?usp=drivesdk):
174,154 five-track piano-rolls derived from midi files in LMD.
- [lpd-5-matched](https://drive.google.com/file/d/1BjjmX_gxStUC45dSaHa-uaICGOVqcy_c/view?usp=drivesdk)\*:
115,160 five-track piano-rolls derived from midi files of 30,886 songs in LMD-matched.
- [lpd-5-cleansed](https://drive.google.com/file/d/1Td86zpOU5ghgARYyeBqsXir5CRljBi_u/view?usp=drivesdk)\*:
24,474 five-track piano-rolls derived from midi files of distinct songs in LMD-matched.

\* These files are matched to entries in the Million Song Dataset (MSD).
To make use of the metadata from MSD, we refer users to the [demo page](http://colinraffel.com/projects/lmd/) of LMD.

## Using LPD

The multi-track piano-rolls in LPD are stored in a special format for efficient I/O and to save space.
Please use [Pypianoroll](https://salu133445.github.io/pypianoroll/) to load the data properly.

### License

Lakh Pianoroll Dataset is a derivative of [Lakh MIDI dataset](http://colinraffel.com/projects/lmd/) by [Colin Raffel](http://colinraffel.com), used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Lakh Pianoroll Dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by [Hao-Wen Dong](https://salu133445.github.io/) and [Wen-Yi Hsiao](https://github.com/wayne391).

Please cite the following papers if you use Lakh Pianoroll Dataset in a published work:

- Hao-Wen Dong, Wen-Yi Hsiao, Li-Chia Yang and Yi-Hsuan Yang,
"MuseGAN: Multi-track Sequential Generative Adversarial Networks for Symbolic Music Generation and Accompaniment,"
in AAAI Conference on Artificial Intelligence (AAAI), 2018.

- Colin Raffel,
"Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching,"
*PhD Thesis*, 2016.
