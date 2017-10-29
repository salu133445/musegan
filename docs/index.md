# MuseGAN

Hao-Wen Dong, Wen-Yi Hsiao, Li-Chia Yang and Yi-Hsuan Yang. 2017. **MuseGAN: Symbolic-domain Music Generation and Accompaniment with Multi-track Sequential Generative Adversarial Networks.** *arXiv preprint arXiv:1709.06298*.
[[arxiv](http://arxiv.org/abs/1709.06298)]

Hao-Wen Dong, Wen-Yi Hsiao, Li-Chia Yang and Yi-Hsuan Yang. 2017.  **MuseGAN: Demonstration of a Convolutional GAN Based Model for Generating Multi-track Piano-rolls.** *ISMIR'17 Late-Breaking and Demo Session*. (non-peer reviewed two-page extended abstract)
[[paper](pdf/musegan-ismir17-lbd.pdf)]

<br><br>

# Introduction

Generating music has a few notable differences from generating images and videos. First, music is an art of time, necessitating a temporal model. Second, music is usually composed of multiple instruments/tracks, with close interaction with one another. Each track has its own temporal dynamics, but collectively they unfold over time interdependently. Lastly, for symbolic domain music generation, the targeted output is sequences of discrete musical events, not continuous values. In this paper, we propose and study three generative adversarial networks (GANs) for symbolic-domain multi-track music generation, using a data set of 127,731 MIDI bars of pop/rock music. The three models, which differ in the underlying model assumption and accordingly the network architecture, are referred to as the jamming model, composer model, and hybrid model, respectively. We propose a few intra-track and inter-track objective metrics to examine and compare their generation result, in addition to a subjective evaluation. We show that our models can learn from the noisy MIDI files and generate coherent music of four bars right from scratch (i.e. without human inputs). We also propose extensions of our models to facilitate human-AI cooperative music creation: given the piano track composed by human we can generate four additional tracks in return to accompany it.

![musegan](fig/musegan.png "musegan")
<p align="center">System diagram of MuseGAN</p>


# Data

## What is a piano-roll?

Piano-roll is a music storing format which represents a music piece by a score-like matrix. The vertical axis represents note pitch, and the horizontal axis represents time. The time axis can be either in absolute timing or in symbolic timing. With absolute timing, the actual timing of note ocurrence is used. With symbolic timing, each beat has the same length. The tempo information are removed.

In our work, we use symbolic timing and we set the time resolution of a beat to 24 to cover common temporal patterns such as triplets and 16th notes. The note pitch has 84 possibillities, covering from C1 to C8 (excluded). For example, a bar in 4/4 time can be represented as a 96 x 84 piano-roll matrix.

Note that a piano-roll is typically single-track. We simply represents a multi-track music piece with a *multi-track piano-roll*, which is a set of piano-rolls where each piano-roll represents one specific track of the original music piecce. That is, a N-track music piece will be converted into a set of N piano-rolls.

## Lakh Piano-roll Dataset (LPD)

The Lakh Piano-roll Dataset (LPD) is a collection of 173,997 unique multi-track piaorolls derived from midi files in the Lakh MIDI Dataset (LMD). More infomaton about the LMD, please go to the [demo page](http://colinraffel.com/projects/lmd/) of LMD.

- [lpd-matched](https://drive.google.com/file/d/0Bx-qnQlE_EmsWG1LbVY0MHY5ems/view?usp=drivesdk): 115006 midi files from 30887 songs converted into multi-track piano-rolls. These files are matched to entries in the Million Song Dataset (MSD). To make use of the metadata from MSD, we refer the user to the [demo page](http://colinraffel.com/projects/lmd/) of LMD.

- [lpd-full](https://drive.google.com/file/d/0Bx-qnQlE_EmseEtIWGR6WHVoQmM/view?usp=drivesdk): 173997 midi files converted into multi-track piano-rolls.

- [sparse_npz.py](https://drive.google.com/open?id=0Bx-qnQlE_EmsMFRISEd2MFJsS3c): utilities for saving/loading multiple scipy.sparse.csc_matrix in one npz file.

Please use the provided utilities to load the npz files directly into csc_matrices.

### License/attribution

Lakh Piano-roll Dataset is a derivative of [Lakh MIDI dataset](http://colinraffel.com/projects/lmd/) by [Colin Raffel](http://colinraffel.com), used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Lakh Piano-roll Dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by Hao-Wen Dong and Wen-Yi Hsiao.

Please reference the following papers if you use this dataset.

- Colin Raffel. "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching". PhD Thesis, 2016.

- Hao-Wen Dong, Wen-Yi Hsiao, Li-Chia Yang, Yi-Hsuan Yang. "MuseGAN: Symbolic-domain Music Generation and Accompaniment with Multi-track Sequential Generative Adversarial Networks". arXiv preprint arXiv:1709.06298. 2017.


# Results

![evolution](fig/evolution.png "evolution")
<p align="center">Evolution of the generated piano-rolls as a function of update steps</p>

![hybrid](fig/hybrid.png "hybrid")
<p align="center">Randomly-picked generation result (piano-rolls), generating from scratch</p>


# Audio Samples
## Best Samples
{% include audio.html filename="composer_best.mp3" %}
<!-- {% include player.html id="342139222" token="s-ucrbd" %} -->

## Generation From Scratch
- the *jamming* model
{% include player.html id="342139219" token="s-AtsSW" %}

- the *composer* model
{% include player.html id="342139221" token="s-pmUd2" %}

- the *hybrid* model
{% include player.html id="342139220" token="s-XH64Z"%}

## Track-conditional Generation
- the *jamming* model
{% include player.html id="342139215" token="s-414n1" %}

- the *composer* model
{% include player.html id="342139218" token="s-HaAWz" %}

- the *hybrid* model
{% include player.html id="342139217" token="s-zC6QN" %}