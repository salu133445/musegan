# Dataset

## Lakh Pianoroll Dataset (LPD)

The Lakh Pianoroll Dataset (LPD) is a collection of 173,997 unique multi-track piano-rolls derived from MIDI files in the Lakh MIDI Dataset (LMD). More information about the LMD, please go to the [demo page](http://colinraffel.com/projects/lmd/) of the LMD.

- [lpd-full](https://drive.google.com/file/d/0Bx-qnQlE_EmseEtIWGR6WHVoQmM/view?usp=drivesdk): 173997 multi-track piano-rolls converted from midi files in LMD.
- [lpd-matched](https://drive.google.com/file/d/0Bx-qnQlE_EmsWG1LbVY0MHY5ems/view?usp=drivesdk)\*: 115006 multi-track piano-rolls converted from midi files of 30887 songs in LMD-matched.
- [lpd-cleansed](https://drive.google.com/file/d/1Sw3VQL8c-v6rw2qii0fh1h3ItQ1oOfKx/view?usp=drivesdk)\*: 21240 multi-track piano-rolls converted from midi files of distinct songs in LMD-matched.

- [lpd-5-full](https://drive.google.com/file/d/1ml_2idsl2ZNdg96BiuGcPRZVyUyKoyhs/view?usp=drivesdk): 173997 five-track piano-rolls converted from midi files in LMD.
- [lpd-5-matched](https://drive.google.com/file/d/1bwZO7MV9t1LGvFMI3gcQdqfzzj-6VW_I/view?usp=drivesdk)\*: 115006 five-track piano-rolls converted from midi files of 30887 songs in LMD-matched.
- [lpd-5-cleansed](https://drive.google.com/file/d/1Mf4mcPr_98nszVoTEXc-g_O0jsniWDCr/view?usp=drivesdk)\*: 21240 five-track piano-rolls converted from midi files of distinct songs in LMD-matched.

Please use the following utilities to load the npz files directly into csc_matrices.

- [sparse_npz.py](https://drive.google.com/open?id=0Bx-qnQlE_EmsMFRISEd2MFJsS3c): utilities for saving/loading multiple scipy.sparse.csc_matrix in one npz file.

\* These files are matched to entries in the Million Song Dataset (MSD). To make use of the metadata from MSD, we refer users to the [demo page](http://colinraffel.com/projects/lmd/) of the LMD.

### License/attribution

Lakh Pianoroll Dataset is a derivative of [Lakh MIDI dataset](http://colinraffel.com/projects/lmd/) by [Colin Raffel](http://colinraffel.com), used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Lakh Pianoroll Dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by [Hao-Wen Dong](https://salu133445.github.io/) and [Wen-Yi Hsiao](https://github.com/wayne391).

Please reference the following papers if you use this dataset.

- Colin Raffel, "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching," *PhD Thesis*, 2016.

- Hao-Wen Dong, Wen-Yi Hsiao, Li-Chia Yang and Yi-Hsuan Yang, "MuseGAN: Multi-track Sequential Generative Adversarial Networks for Symbolic Music Generation and Accompaniment," in AAAI Conference on Artificial Intelligence (AAAI), 2018.
