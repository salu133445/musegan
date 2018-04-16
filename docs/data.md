# Data Representation

## Piano-roll

Piano-roll is a music storing format which represents a music piece by a
score-like matrix. The vertical and horizontal axes represent note pitch and
time, respectively. The values represent the velocities of the notes.

The time axis can either be in *absolute timing* or in *symbolic timing*. For
absolute timing, the actual timing of note occurrence is used. For symbolic
timing, the tempo information is removed and thereby each beat has the same
length.

In our work, we use symbolic timing and the temporal resolution is set to 24 per
beat in order to cover common temporal patterns such as triplets and 32th notes.
The note pitch has 128 possibilities, covering from C-1 to G9. For example, a
bar in 4/4 time with only one track can be represented as a 96 x 84 matrix.

> Note that during the conversion from MIDI files to piano-rolls, an additional
minimal-length (of one time step) pause is added between two consecutive
(without a pause) notes of the same pitch to distinguish them from one single
note.

<img src="figs/pianoroll-example.png" alt="pianoroll-example" style="max-height:200px; display:block; margin:auto">
<p class="caption" align="center">Example piano-roll</p>

## Multi-track Piano-roll

We represent a multi-track music piece with a *multi-track piano-roll*, which is
a set of piano-rolls where each piano-roll represents one specific track in the
original music piece. That is, a *M*-track music piece will be converted into a
set of *M* piano-rolls. For instance, a bar in 4/4 time with *M* tracks can be
represented as a 96 x 84 x *M* tensor.

<img src="figs/pianoroll-example-5tracks.png" alt="pianoroll-example-5tracks" style="max-height:400px; display:block; margin:auto">
<p class="caption" align="center">Example five-track piano-rolls</p>

*The above piano-roll visualizations are produced using
[Pypianoroll](https://salu133445.github.io/pypianoroll/).*