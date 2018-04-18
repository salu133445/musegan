# Abstract

Generating music has a few notable differences from generating images and
videos. First, music is an art of time, necessitating a temporal model. Second,
music is usually composed of multiple instruments/tracks with their own temporal
dynamics, but collectively they unfold over time interdependently. Lastly,
musical notes are often grouped into chords, arpeggios or melodies in polyphonic
music, and thereby introducing a chronological ordering of notes is not
naturally suitable. In this paper, we propose three models for symbolic
multi-track music generation under the framework of generative adversarial
networks (GANs). The three models, which differ in the underlying assumptions
and accordingly the network architectures, are referred to as the jamming model,
the composer model and the hybrid model. We trained the proposed models on a
dataset of over one hundred thousand bars of rock music and applied them to
generate piano-rolls of five tracks: bass, drums, guitar, piano and strings. A
few intra-track and inter-track objective metrics are also proposed to evaluate
the generative results, in addition to a subjective user study. We show that our
models can generate coherent music of four bars right from scratch (i.e. without
human inputs). We also extend our models to human-AI cooperative music
generation: given a specific track composed by human, we can generate four
additional tracks to accompany it.
