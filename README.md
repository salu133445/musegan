# MuseGAN

[MuseGAN](https://salu133445.github.io/musegan/) is a project on music
generation. In a nutshell, we aim to generate polyphonic music of multiple
tracks (instruments). The proposed models are able to generate music either from
scratch, or by accompanying a track given a priori by the user.

We train the model with training data collected from
[Lakh Pianoroll Dataset](https://salu133445.github.io/lakh-pianoroll-dataset/)
to generate pop song phrases consisting of bass, drums, guitar, piano and
strings tracks.

Sample results are available
[here](https://salu133445.github.io/musegan/results).

## Important Notes

- The latest implementation is based on the network architectures presented in BinaryMuseGAN, where the temporal structure is handled by 3D convolutional layers. The advantage of this design is its smaller network size, while the disadvantage is its reduced controllability, e.g., capability of feeding different latent variables for different measures or tracks.
- The original code we used for running the experiments in the paper can be found in the `v1` folder.
- Looking for a PyTorch version? Check out [this repository](https://github.com/salu133445/ismir2019tutorial).

## Prerequisites

> __Below we assume the working directory is the repository root.__

### Install dependencies

- Using pipenv (recommended)

  > Make sure `pipenv` is installed. (If not, simply run `pip install pipenv`.)

  ```sh
  # Install the dependencies
  pipenv install
  # Activate the virtual environment
  pipenv shell
  ```

- Using pip

  ```sh
  # Install the dependencies
  pip install -r requirements.txt
  ```

### Prepare training data

> The training data is collected from
[Lakh Pianoroll Dataset](https://salu133445.github.io/lakh-pianoroll-dataset/)
(LPD), a new multitrack pianoroll dataset.

```sh
# Download the training data
./scripts/download_data.sh
# Store the training data to shared memory
./scripts/process_data.sh
```

You can also download the training data manually
([train_x_lpd_5_phr.npz](https://docs.google.com/uc?export=download&id=14rrC5bSQkB9VYWrvt2IhsCjOKYrguk3S)).

> As pianoroll matrices are generally sparse, we store only the indices of
nonzero elements and the array shape into a npz file to save space, and later
restore the original array. To save some training data `data` into this format,
simply run
`np.savez_compressed("data.npz", shape=data.shape, nonzero=data.nonzero())`

## Scripts

We provide several shell scripts for easy managing the experiments. (See
[here](scripts/README.md) for a detailed documentation.)

> __Below we assume the working directory is the repository root.__

### Train a new model

1. Run the following command to set up a new experiment with default settings.

   ```sh
   # Set up a new experiment
   ./scripts/setup_exp.sh "./exp/my_experiment/" "Some notes on my experiment"
   ```

2. Modify the configuration and model parameter files for experimental settings.

3. You can either train the model:

     ```sh
     # Train the model
     ./scripts/run_train.sh "./exp/my_experiment/" "0"
     ```

   or run the experiment (training + inference + interpolation):

     ```sh
     # Run the experiment
     ./scripts/run_exp.sh "./exp/my_experiment/" "0"
     ```

### Collect training data

Run the following command to collect training data from MIDI files.

  ```sh
  # Collect training data
  ./scripts/collect_data.sh "./midi_dir/" "data/train.npy"
  ```

### Use pretrained models

1. Download pretrained models

   ```sh
   # Download the pretrained models
   ./scripts/download_models.sh
   ```

   You can also download the pretrained models manually
   ([pretrained_models.tar.gz](https://docs.google.com/uc?export=download&id=19RYAbj_utCDMpU7PurkjsH4e_Vy8H-Uy)).

2. You can either perform inference from a trained model:

   ```sh
   # Run inference from a pretrained model
   ./scripts/run_inference.sh "./exp/default/" "0"
   ```

   or perform interpolation from a trained model:

   ```sh
   # Run interpolation from a pretrained model
   ./scripts/run_interpolation.sh "./exp/default/" "0"
   ```

## Outputs

By default, samples will be generated alongside the training. You can disable
this behavior by setting `save_samples_steps` to zero in the configuration file
(`config.yaml`). The generated will be stored in the following three formats by
default.

- `.npy`: raw numpy arrays
- `.png`: image files
- `.npz`: multitrack pianoroll files that can be loaded by the
  _[Pypianoroll](https://salu133445.github.io/pypianoroll/index.html)_
  package

You can disable saving in a specific format by setting `save_array_samples`,
`save_image_samples` and `save_pianoroll_samples` to `False`  in the
configuration file.

The generated pianorolls are stored in .npz format to save space and processing
time. You can use the following code to write them into MIDI files.

```python
from pypianoroll import Multitrack

m = Multitrack('./test.npz')
m.write('./test.mid')
```

## Sample Results

Some sample results can be found in `./exp/` directory. More samples can be
downloaded from the following links.

- [`sample_results.tar.gz`](https://docs.google.com/uc?export=download&id=1BsNtc8_mpLK5l2F5jncIkHbTcJqtZu2w) (54.7 MB):
  sample inference and interpolation results
- [`training_samples.tar.gz`](https://docs.google.com/uc?export=download&id=1pZk0YCElcHHSBfhbV8j_zaRr1zhEQUzN) (18.7 MB):
  sample generated results at different steps

Citing
------

Please cite the following paper if you use the code provided in this repository.

Hao-Wen Dong,\* Wen-Yi Hsiao,\* Li-Chia Yang and Yi-Hsuan Yang, "MuseGAN: Multi-track Sequential Generative Adversarial Networks for Symbolic
Music Generation and Accompaniment," _Proceedings of the 32nd AAAI Conference on Artificial Intelligence (AAAI)_, 2018. (\*equal contribution)
<br>
[[homepage](https://salu133445.github.io/musegan)]
[[arXiv](http://arxiv.org/abs/1709.06298)]
[[paper](https://salu133445.github.io/musegan/pdf/musegan-aaai2018-paper.pdf)]
[[slides](https://salu133445.github.io/musegan/pdf/musegan-aaai2018-slides.pdf)]
[[code](https://github.com/salu133445/musegan)]

## Papers

__MuseGAN: Multi-track Sequential Generative Adversarial Networks for Symbolic Music Generation and Accompaniment__<br>
Hao-Wen Dong,\* Wen-Yi Hsiao,\* Li-Chia Yang and Yi-Hsuan Yang (\*equal contribution)<br>
_Proceedings of the 32nd AAAI Conference on Artificial Intelligence (AAAI)_, 2018.<br>
[[homepage](https://salu133445.github.io/musegan)]
[[arXiv](http://arxiv.org/abs/1709.06298)]
[[paper](https://salu133445.github.io/musegan/pdf/musegan-aaai2018-paper.pdf)]
[[slides](https://salu133445.github.io/musegan/pdf/musegan-aaai2018-slides.pdf)]
[[code](https://github.com/salu133445/musegan)]

__Convolutional Generative Adversarial Networks with Binary Neurons for Polyphonic Music Generation__<br>
Hao-Wen Dong and Yi-Hsuan Yang<br>
_Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR)_, 2018.<br>
[[homepage](https://salu133445.github.io/bmusegan)]
[[video](https://youtu.be/r9C2Q2oR9Ik)]
[[paper](https://salu133445.github.io/bmusegan/pdf/bmusegan-ismir2018-paper.pdf)]
[[slides](https://salu133445.github.io/bmusegan/pdf/bmusegan-ismir2018-slides.pdf)]
[[slides (long)](https://salu133445.github.io/bmusegan/pdf/bmusegan-tmac2018-slides.pdf)]
[[poster](https://salu133445.github.io/bmusegan/pdf/bmusegan-ismir2018-poster.pdf)]
[[arXiv](https://arxiv.org/abs/1804.09399)]
[[code](https://github.com/salu133445/bmusegan)]

__MuseGAN: Demonstration of a Convolutional GAN Based Model for Generating Multi-track Piano-rolls__<br>
Hao-Wen Dong,\* Wen-Yi Hsiao,\* Li-Chia Yang and Yi-Hsuan Yang (\*equal contribution)<br>
_Late-Breaking Demos of the 18th International Society for Music Information Retrieval Conference (ISMIR)_, 2017.<br>
[[paper](https://salu133445.github.io/musegan/pdf/musegan-ismir2017-lbd-paper.pdf)]
[[poster](https://salu133445.github.io/musegan/pdf/musegan-ismir2017-lbd-poster.pdf)]
