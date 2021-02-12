# Shell scripts

We provide several shell scripts for easy managing the experiments.

| File                   | Description                                                |
|------------------------|------------------------------------------------------------|
| `download_data.sh`     | Download the training data                                 |
| `generate_data.sh`     | Generate data from a set of midi files                     |
| `process_data.sh`      | Save the training data to shared memory                    |
| `download_models.sh`   | Download the pretrained models                             |
| `setup_exp.sh`         | Set up a new experiment with default settings              |
| `run_train.sh`         | Train a model                                              |
| `run_inference.sh`     | Run inference from a trained model                         |
| `run_interpolation.sh` | Run interpolation from a trained model                     |
| `run_exp.sh`           | Run an experiment (training + inference + interpolation)   |
| `rerun_exp.sh`         | Rerun an experiment (training + inference + interpolation) |

> __Below we assume the working directory is the repository root.__

## Download the training data

```sh
./scripts/download_data.sh
```

This command will download the training data to the default data directory
(`./data/`).

## Generate a dataset from a set of midi files

```sh
./scripts/generate_data.sh "./music_dir/"
```

This command will generate training data from a given directory (`./music_dir/`) 
by looking for all the files in that directory that end wih `.mid` and converting
them to a five track pianoroll dataset.

## Save the training data to shared memory

```sh
./scripts/process_data.sh
```

This command will store the training data to shared memory using SharedArray
package.

## Download the pretrained models

```sh
./scripts/download_models.sh
```

This command will download the pretrained models to the default experiment
directory (`./exp/`).

## Set up a new experiment with default settings

```sh
./scripts/setup_exp.sh "./exp/my_experiment/" "Some notes"
```

This command will create a new experiment directory at the given path
(`./exp/my_experiment/`), copy the default configuration and model parameter
files to that folder and save the experiment note (`"Some notes"`) as a text
file in that folder.

## Train a model

```sh
./scripts/run_train.sh "./exp/my_experiment/" "0"
```

This command will look for the configuration and model parameter files in the
given experiment directory (`./exp/my_experiment/`) and train a model according
to the configurations and parameters on the specified GPU (`"0"`).

## Run inference from a trained model

```sh
./scripts/run_inference.sh "./exp/my_experiment/" "0"
```

This command will look for the configuration and model parameter files in the
given experiment directory (`./exp/my_experiment/`) and run inference from the
trained model on the specified GPU (`"0"`).

## Run interpolation from a trained model

```sh
./scripts/run_interpolation.sh "./exp/my_experiment/" "0"
```

This command will look for the configuration and model parameter files in the
given experiment directory (`./exp/my_experiment/`) and run interpolation from the
trained model on the specified GPU (`"0"`).

## Run an experiment (train + inference + interpolation)

We provide a simple wrapper for a typical experiment scenario&mdash;first we
train a model and then we run inference and interpolation from the trained
model. In this case, you can simply run the following command.

```sh
./scripts/run_exp.sh "./exp/my_experiment/" "0"
```

This command will look for the configuration and model parameter files in the
given experiment directory (`./exp/my_experiment/`) and run the experiment
(train, inference and interpolation) on the specified GPU (`"0"`).

## Rerun an experiment (train + inference + interpolation)

```sh
./scripts/rerun_exp.sh "./exp/my_experiment/" "0"
```

This command will remove everything in the experiment directory except the
configuration and model parameter files and then rerun the experiment (train,
inference and interpolation) on the specified GPU (`"0"`).
