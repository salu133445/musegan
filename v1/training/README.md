# Source code for preparing the training data

> Note that the source code is not well-organized. But we think simply releasing
the code can still be useful.

## Step 1 - Collect qualified piano-rolls

```console
python tracks_parsing.py
```

Given a list of qualified MSD ids (`Rock_C_id` here), this file merge the data
into five tracks: *Bass*, *Drums*, *Guitar*, *Piano* and *Strings*. Instruments
out of the five categories are considered as part of the strings.

- Multiple tracks in LMD => {Drum, Piano, Guitar, Bass, Others}
- 2844 selected in 6646

The collected piano-rolls are placed in `tracks/` folder with names
"[instrument_name]/[msd_id].npz", e.g. `Drum/TRAAEEH128E0795DFE.npz`.

## Step 2 - Prepare files for structure analysis

```console
python "song_analysis.py"
```

This python file does the following.

- Convert piano-rolls to .mat format
  - `Piano_Roll.mat`: matrix for running segmentation in MATLAB. The shape is
    (128, T).

- Create activation labels
  - `act_instr/[msd_id].npy`: 2D numpy array of shape (num_bar, 5). Indicate
    whether the corresponding instrument is activated (non-empty) in each bar.
  - `act_all/[msd_id].npy`: list of bool. Indicate whether the number of
    activated instruments is larger than a threshold (3 here).

The results are placed in the `trakcs/` folder.

## Step 3 - Segment and label the collected piano-rolls

```console
matlab main_seg.m
matlab main_lab.m
```

These files perform the segmentation and labeling, respectively. The algorithm
we employed is called *Structural Feature* [1, 2]. Although it is designed for
raw audios, we found it works well on symbolic data (i.e. piano-rolls) as well.
For python user, you can use [MSAF](https://github.com/urinieto/msaf) to perform
structure analysis as well.

*The code is modified from Wayne's previous project &mdash;
[Music-Structure-Analysis-in-Matlab](https://github.com/wayne391/Music-Structure-Analysis-in-Matlab) [3].*

The analysis results are placed in the `structure/` folder.

## Step 4 - Collect the training data

```console
python gen_data_bar.py
```

This file collect the training data for the five tracks and generate an
additional `act_instr` track which indicates the instrument activations.
The training data is filtered by `act_all` and collected following some rules
on top of the analysis results obtained in Step 3.

## References

1. Joan Serrá, Meinard Müller, Peter Grosche and Josep Ll. Arcos,
   "**Unsupervised Detection of Music Boundaries by Time Series Structure
   Features**,
   in *AAAI Conference on Artificial Intelligence* (AAAI), 2012

2. Joan Serrá, Meinard Müller, Peter Grosche and Josep Ll. Arcos,
   "**Unsupervised Music Structure Annotation by Time Series Structure Features and
   Segment Similarity**",
   in *IEEE Transactions on Multimedia*, vol. 16, no. 5, pp. 1229-1240, 2014.

3. https://github.com/wayne391/Music-Structure-Analysis-in-Matlab
