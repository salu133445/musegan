# Model

## Modeling Multi-track Interdependency

In our experience, there are two common ways to create music.

- Given a group of musicians playing different instruments, they can create
  music by improvising music without a predefined arrangement, a.k.a. jamming.
- A composer arranges instruments with knowledge of harmonic structure and
  instrumentation. Musicians will then follow the composition and play the
  music.

We design three models corresponding to these compositional approaches.

### Jamming Model

Multiple generators work independently and generate music of its own track from
a private random vector <i>z<sub>i</sub></i> (*i* = 1 &hellip; *M*), where *M*
denotes the number of generators (or tracks). These generators receive critics
(i.e. backpropogated supervisory signals) from different discriminators.

<img src="figs/multitrack1.png" alt="jamming_model" style="width:100%; max-width:300px; display:block;">
<p class="caption">Jamming model</p>

### Composer Model

One single generator creates a multi-channel piano-roll, with each channel
representing a specific track. This model requires only one shared random vector
*z* (which may be viewed as the intention of the composer) and one
discriminator, which examines the *M* tracks collectively to tell whether the
input music is real or fake.

<img src="figs/multitrack2.png" alt="composer_model" style="width:100%; max-width:300px; display:block;">
<p class="caption">Composer model</p>

### Hybrid Model

Combining the idea of jamming and composing, the hybrid model require *M*
generators and each takes as inputs an inter-track random vector *z* and an
intra-track random vector <i>z<sub>i</sub></i>. We expect that the inter-track
random vector can coordinate the generation of different musicians, namely
<i>G<sub>i</sub></i>, just like a composer does. Moreover, we use only one
discriminator to evaluate the *M* tracks collectively.

> A major difference between the composer model and the hybrid model lies in the
flexibility&mdash;in the hybrid model we can use different network architectures
(e.g., number of layers, filter size) and different inputs for the *M*
generators. Therefore, we can for example vary the generation of one specific
track without losing the inter-track interdependency.

<img src="figs/multitrack3.png" alt="hybrid_model" style="width:100%; max-width:300px; display:block;">
<p class="caption">Hybrid model</p>

## Modeling Temporal Structure

### Generation from Scratch

The first method aims to generate fixed-length musical phrases by viewing bar
progression as another dimension to grow the generator. The generator consists
of two sub networks, the *temporal structure generator* <i>G<sub>temp</sub></i>
and the *bar generator* <i>G<sub>bar</sub></i>. <i>G<sub>temp</sub></i> maps a
noise vector to a sequence of some latent vectors, which is expected to carry
temporal information and used by <i>G<sub>bar</sub></i> to generate
piano-rolls sequentially (i.e. bar by bar).

<img src="figs/temporal1.png" alt="from_scratch_model" style="width:100%; max-width:300px; display:block;">
<p class="caption">Generation from Scratch</p>

### Track-conditional Generation

The second method assumes that the bar sequence of one specific track is given
by human, and tries to learn the temporal structure underlying that track and
to generate the remaining tracks (and complete the song). The track-conditional
generator *G*&deg; generates bars one after another with the *conditional bar
generator*, *G*&deg;<i><sub>bar</sub></i>, which takes as inputs the conditional
track and a random noise. In order to achieve such conditional generation with
high-dimensional conditions, an additional encoder *E* is trained to map the
condition to the space of *z*.

> Note that the encoder is expected to extract inter-track features instead of
intra-track features from the given track, since intra-track features are
supposed not to be useful for generating the other tracks.

<img src="figs/temporal2.png" alt="track_conditional_model" style="width:100%; max-width:300px; display:block;">
<p class="caption">Track-conditional Generation</p>

## MuseGAN

MuseGAN, an integration and extension of the proposed multi-track and temporal
models, takes as input four different types of random vectors:

- an *inter-track time-independent* random vector (*z*)
- an *intra-track time-independent* random vector (<i>z<sub>i</sub></i>)
- *inter-track time-dependent* random vectors (<i>z<sub>t</sub></i>)
- *intra-track time-dependent* random vectors (<i>z<sub>i, t</sub></i>)

For track *i* (*i* = 1 &hellip; *M*), the *shared* temporal structure generator
<i>G<sub>temp</sub></i> and the *private* temporal structure generator
<i>G<sub>temp, i</sub></i> take the time-dependent random vectors,
<i>z<sub>t</sub></i> and <i>z<sub>i, t</sub></i>, respectively, as their inputs,
and each of them outputs a series of latent vectors containing inter-track and
intra-track, respectively, temporal information. The output series (of latent
vectors), together with the time-independent random vectors, *z* and
<i>z<sub>i</sub></i>, are concatenated and fed to the bar generator
<i>G<sub>bar</sub></i>, which then generates piano-rolls sequentially.

![musegan](figs/musegan.png)
