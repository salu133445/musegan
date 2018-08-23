# MuseGAN

<font color=red><strong><i>Warning: this version is no longer maintained</i></strong></font>

[MuseGAN](https://salu133445.github.io/musegan/) is a project on music
generation. In essence, we aim to generate polyphonic music of multiple tracks
(instruments) with harmonic and rhythmic structure, multi-track interdependency
and temporal structure. To our knowledge, our work represents the first approach
that deal with these issues altogether.

The models are trained with
[Lakh Pianoroll Dataset](https://salu133445.github.io/lakh-pianoroll-dataset/)
(LPD), a new [multi-track piano-roll](https://salu133445.github.io/musegan/data)
dataset, in an unsupervised approach. The proposed models are able to generate
music either from scratch, or by accompanying a track given by user.
Specifically, we use the model to generate pop song phrases consisting of bass,
drums, guitar, piano and strings tracks.

Sample results are available [here](https://salu133445.github.io/musegan/results).

## Papers

Hao-Wen Dong\*, Wen-Yi Hsiao\*, Li-Chia Yang and Yi-Hsuan Yang,
"**MuseGAN: Multi-track Sequential Generative Adversarial Networks for
Symbolic Music Generation and Accompaniment**,"
in *AAAI Conference on Artificial Intelligence* (AAAI), 2018.
[[arxiv](http://arxiv.org/abs/1709.06298)]
[[slides](https://salu133445.github.io/musegan/pdf/musegan-aaai2018-slides.pdf)]

Hao-Wen Dong\*, Wen-Yi Hsiao\*, Li-Chia Yang and Yi-Hsuan Yang,
"**MuseGAN: Demonstration of a Convolutional GAN Based Model for Generating
Multi-track Piano-rolls**,"
in *ISMIR Late-Breaking and Demo Session*, 2017.
(non-peer reviewed two-page extended abstract)
[[paper](https://salu133445.github.io/musegan/pdf/musegan-ismir2017-lbd-paper.pdf)]
[[poster](https://salu133445.github.io/musegan/pdf/musegan-ismir2017-lbd-poster.pdf)]

\* *These authors contributed equally to this work.*

## Usage

```python
import tensorflow as tf
from musegan.core import MuseGAN
from musegan.components import NowbarHybrid
from config import *

# Initialize a tensorflow session
with tf.Session() as sess:

    # === Prerequisites ===
    # Step 1 - Initialize the training configuration
    t_config = TrainingConfig

    # Step 2 - Select the desired model
    model = NowbarHybrid(NowBarHybridConfig)

    # Step 3 - Initialize the input data object
    input_data = InputDataNowBarHybrid(model)

    # Step 4 - Load training data
    path_train = 'train.npy'
    input_data.add_data(path_train, key='train')

    # Step 5 - Initialize a museGAN object
    musegan = MuseGAN(sess, t_config, model)

    # === Training ===
    musegan.train(input_data)

    # === Load a Pretrained Model ===
    musegan.load(musegan.dir_ckpt)

    # === Generate Samples ===
    path_test = 'train.npy'
    input_data.add_data(path_test, key='test')
    musegan.gen_test(input_data, is_eval=True)
```
