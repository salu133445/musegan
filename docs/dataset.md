# Dataset

## Lakh Pianoroll Dataset (LPD)

The Lakh Pianoroll Dataset (LPD) is a collection of 173,997 unique multi-track piano-rolls derived from MIDI files in the Lakh MIDI Dataset (LMD). More information about the LMD, please go to the [demo page](http://colinraffel.com/projects/lmd/) of the LMD.

- [lpd-matched](https://drive.google.com/file/d/0Bx-qnQlE_EmsWG1LbVY0MHY5ems/view?usp=drivesdk): 115006 midi files from 30887 songs converted into multi-track piano-rolls. These files are matched to entries in the Million Song Dataset (MSD). To make use of the metadata from MSD, we refer users to the [demo page](http://colinraffel.com/projects/lmd/) of the LMD.

- [lpd-full](https://drive.google.com/file/d/0Bx-qnQlE_EmseEtIWGR6WHVoQmM/view?usp=drivesdk): 173997 midi files converted into multi-track piano-rolls.

- [sparse_npz.py](https://drive.google.com/open?id=0Bx-qnQlE_EmsMFRISEd2MFJsS3c): utilities for saving/loading multiple scipy.sparse.csc_matrix in one npz file.

Please use the provided utilities to load the npz files directly into csc_matrices.

### License/attribution

Lakh Pianoroll Dataset is a derivative of [Lakh MIDI dataset](http://colinraffel.com/projects/lmd/) by [Colin Raffel](http://colinraffel.com), used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Lakh Pianoroll Dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by Hao-Wen Dong and Wen-Yi Hsiao.

Please reference the following papers if you use this dataset.

- Colin Raffel. "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching". PhD Thesis, 2016.

- Hao-Wen Dong, Wen-Yi Hsiao, Li-Chia Yang, Yi-Hsuan Yang. "MuseGAN: Symbolic-domain Music Generation and Accompaniment with Multi-track Sequential Generative Adversarial Networks". arXiv preprint arXiv:1709.06298. 2017.