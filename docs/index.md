
<img src="figs/logo.png" alt="logo" width="200" height="200" />

(accepted by AAAI2018)

MuseGAN is a project on multi-track sequential data generation. In essence, we aim to generate multi-track polyphonic music with harmonic and rhythmic structure, multi-track interdependency and temporal structure from a large collection of MIDIs in an unsupervised approach. To our knowledge, our work represents the first approach that deal with these issues altogether.

Our model can generate music either from scratch, or by accompanying a track given by user. Specifically, we use the model to generate pop song segments consisting of bass, drums, guitar, piano and strings tracks.

{% include player.html filename="best_samples.mp3" %}

![musegan](figs/musegan.png)
<p align="center">System diagram of MuseGAN</p>
