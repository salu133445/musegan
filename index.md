# MuseGAN
**MuseGAN: Symbolic-domain Music Generation and Accompaniment with Multi-track Sequential Generative Adversarial Networks**  
[arxiv] https://arxiv.org/abs/XXXX.XXXX
<br><br>  

# Introduction
## Abstract
Generating music has a few notable differences from generating images and videos. First, music is an art of time, necessitating a temporal model. Second, music is usually composed of multiple instruments/tracks, with close interaction with one another. Each track has its own temporal dynamics, but collectively they unfold over time interdependently. Lastly, for symbolic domain music generation, the targeted output is sequences of discrete musical events, not continuous values. In this paper, we propose and study three generative adversarial networks (GANs) for symbolic-domain multi-track music generation, using a data set of 127,731 MIDI bars of pop/rock music. The three models, which differ in the underlying model assumption and accordingly the network architecture, are referred to as the jamming model, composer model, and hybrid model, respectively. We propose a few intra-track and inter-track objective metrics to examine and compare their generation result, in addition to a subjective evaluation. We show that our models can learn from the noisy MIDI files and generate coherent music of four bars right from scratch (i.e. without human inputs). We also propose extensions of our models to facilitate human-AI cooperative music creation: given the piano track composed by human we can generate four additional tracks in return to accompany it.

## System Diagram
![musegan](https://github.com/salu133445/musegan/raw/master/fig/musegan.png "musegan")
<p align="center">System diagram of MuseGAN</p>

## Results

![evolution](https://github.com/salu133445/musegan/raw/master/fig/evolution.png "evolution")
<p align="center">Evolution of the generated piano-rolls as a function of update steps.</p>

![hybrid](https://github.com/salu133445/musegan/raw/master/fig/hybrid.png "hybrid")
<p align="center">Randomly-picked generation result (piano-rolls), generating from scratch.</p>


# Audio Samples
## Best Samples
{% include player.html id="342139222" token="s-ucrbd" %}

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
