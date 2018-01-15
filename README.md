## (In Progress)

Implementation of MuseGAN

Hao-Wen Dong\*, Wen-Yi Hsiao\*, Li-Chia Yang and Yi-Hsuan Yang, "**MuseGAN: Multi-track Sequential Generative Adversarial Networks for Symbolic Music Generation and Accompaniment**," in *AAAI Conference on Artificial Intelligence (AAAI)*, 2018.
[[arxiv](http://arxiv.org/abs/1709.06298)] [[demo](https://salu133445.github.io/musegan/)]

\**These authors contributed equally to this work.*

![image] https://github.com/salu133445/musegan/blob/master/docs/figs/train.gif
### Usage

```python
import tensorflow as tf
from musegan.core import MuseGAN
from musegan.components import NowbarHybrid
from config import *

# initialize a tensorflow session
with tf.Session(config=config) as sess:

    ###  prerequisites ###

    # step 1. initialize the training configuration
    t_config = TrainingConfig

    # step 2. select the desired model
    model = NowbarHybrid(NowBarHybridConfig)

    # step 3. initialize the input data object
    input_data = InputDataNowBarHybrid(model)

    # step 4. load training data
    path_train = 'train.npy'
    input_data.add_data(path_train, key='train')

    # step 5. initialize the museGAN object
    musegan = MuseGAN(sess, t_config, model)



    ###  training ###
    musegan.train(input_data)



    ### load and generate samples ###
    # load pretrained model
    musegan.load(musegan.dir_ckpt)

    # add testing data
    path_test = 'train.npy'
    input_data.add_data(path_test, key='test')

    # generate samples
    musean.gen_test(input_data, is_eval=True)

```

### Progress

- data conversion
- data preprocessing
- data cleansing
- (main code)  *will be released in the near future*
