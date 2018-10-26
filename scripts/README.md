# Shell scripts

We provide several shell scripts for easy managing the experiments.

| File                   | Description                                              |
|------------------------|----------------------------------------------------------|
| `download_data.sh`     | Download the training data                               |
| `process_data.sh`      | Save the training data to shared memory                  |
| `setup_exp.sh`         | Set up a new experiment with default settings            |
| `run_train.sh`         | Train a model                                            |
| `run_inference.sh`     | Run inference from a trained model                       |
| `run_interpolation.sh` | Run interpolation from a trained model                   |
| `run_exp.sh`           | Run an experiment (training + inference + interpolation) |

> __Below we assume the working directory is the repository root.__

## Prepare training data

```sh
# Download the training data
./scripts/download_data.sh
# Store the training data to shared memory
./scripts/process_data.sh
```

> The first command (`./scripts/download_data.sh`) will download the training
data to the default data directory (`[REPO_DIR]/data/`). The second command
(`./scripts/process_data.sh`) will store the training data to shared memory
using SharedArray package.

## Set up a new experiment with default settings

```sh
# Set up a new experiment
./scripts/setup_exp.sh "./exp/my_experiment" "Some notes on my experiment"
```

> This command (`./scripts/setup_exp.sh`) will create a new experiment directory
at the given path (`"./exp/my_experiment"`), copy the default configuration and
model parameter files to that folder and save the experiment note
(`"Some notes on my experiment"`) as a text file in that folder.

## Train a model

```sh
# Train a model
./scripts/run_train.sh "./exp/my_experiment" "0"
```

> The command (`./scripts/run_train.sh`) will look for the configuration and
model parameter files in the given experiment directory
(`"./exp/my_experiment"`) and train a model according to the configurations and
parameters on the specified GPU (`"0"`).

## Perform inference from a trained model

```sh
# Run inference from a pretrained model
./scripts/run_inference.sh "./exp/my_experiment" "0"
```

> The command (`./scripts/run_inference.sh`) will look for the configuration and
model parameter files in the given experiment directory
(`"./exp/my_experiment"`) and perform inference from the trained model on the
specified GPU (`"0"`).

## Perform interpolation from a trained model

```sh
# Run interpolation from a pretrained model
./scripts/run_interpolation.sh "./exp/my_experiment" "0"
```

> The command (`./scripts/run_interpolation.sh`) will look for the configuration
and model parameter files in the given experiment directory
(`"./exp/my_experiment"`) and perform inference from the trained model on the
specified GPU (`"0"`).

## Run an experiment (train + inference + interpolation)

We provide a simple wrapper for a typical experiment scenario&mdash;first we
train a model and then we perform inference and interpolation from the trained
model. In this case, you can simply run the following command.

```sh
# Run an experiment
./scripts/run_exp.sh "./exp/my_experiment" "0"
```

> The command (`./scripts/run_exp.sh`) will look for the configuration and model
parameter files in the given experiment directory (`"./exp/my_experiment"`)
and run the experiment (train, inference and interpolation) on the specified GPU
(`"0"`).

You can also rerun an experiment with the following command.

```sh
# Rerun an experiment
./scripts/rerun_exp.sh "./exp/my_experiment" "0"
```

> The command (`./scripts/run_exp.sh`) will remove everything in the experiment
directory except the configuration and model parameter files and then rerun the
experiment (train, inference and interpolation) on the specified GPU (`"0"`).
