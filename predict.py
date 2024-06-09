import argparse
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from pprint import pformat

import cog
import numpy as np
import scipy.stats
import tensorflow as tf
from pypianoroll import Multitrack

sys.path.append("src")

from musegan.config import LOG_FORMAT, LOGLEVEL
from musegan.data import get_samples, load_data
from musegan.model import Model
from musegan.utils import load_yaml, make_sure_path_exists, update_not_none


def parse_arguments():
    """Parse and return the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", help="Directory where the results are saved.")
    parser.add_argument("--checkpoint_dir", help="Directory that contains checkpoints.")
    parser.add_argument(
        "--params",
        "--params_file",
        "--params_file_path",
        help="Path to the file that defines the " "hyperparameters.",
    )
    parser.add_argument("--config", help="Path to the configuration file.")
    parser.add_argument(
        "--runs", type=int, default="1", help="Times to run the inference process."
    )
    parser.add_argument(
        "--rows", type=int, default=5, help="Number of images per row to be generated."
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=5,
        help="Number of images per column to be generated.",
    )
    parser.add_argument(
        "--lower",
        type=float,
        default=-2,
        help="Lower bound of the truncated normal random " "variables.",
    )
    parser.add_argument(
        "--upper",
        type=float,
        default=2,
        help="Upper bound of the truncated normal random " "variables.",
    )
    parser.add_argument(
        "--gpu",
        "--gpu_device_num",
        type=str,
        default="0",
        help="The GPU device number to use.",
    )
    args = parser.parse_args([])
    return args


class Predictor(cog.Predictor):
    def setup(self):
        """Load the model"""
        # Setup

    @cog.input("seed", type=int, default=-1, help="Random seed, -1 for random")
    @cog.input(
        "sampling_type",
        type=str,
        default="bernoulli_sampling",
        options=["bernoulli_sampling", "hard_thresholding"],
        help="Type of sampling",
    )
    @cog.input(
        "output_type",
        type=str,
        default="audio",
        options=["audio", "midi", "image"],
        help="Type of output",
    )
    def predict(self, seed, sampling_type, output_type):
        """Compute prediction"""
        # set seed
        if seed < 0:
            seed = int.from_bytes(os.urandom(2), "big")
        tf.compat.v1.random.set_random_seed(seed)
        tf.reset_default_graph()  #resolves a bug occuring when running multiple times
        output_dir = Path(tempfile.mkdtemp())
        # output_dir = "prova"
        im_name = (
            "images/fake_x_"
            + sampling_type
            + "_colored/fake_x_"
            + sampling_type
            + "_colored_0.png"
        )
        pianoroll_name = (
            "pianorolls/fake_x_" + sampling_type + "/fake_x_" + sampling_type + "_0.npz"
        )

        output_path_img = output_dir / im_name
        output_path_pianoroll = output_dir / pianoroll_name
        output_path_midi = output_dir / "output.mid"
        output_path_wav = output_dir / "output.wav"
        output_path_mp3 = output_dir / "output.mp3"

        checkpoint_dir = "exp/default/"
        params_file = os.path.join(checkpoint_dir, "params.yaml")
        config_file = os.path.join(checkpoint_dir, "config.yaml")

        args = parse_arguments()
        params = load_yaml(params_file)

        # Load training configurations
        config = load_yaml(config_file)

        update_not_none(config, vars(args))
        config["checkpoint_dir"] = os.path.join(checkpoint_dir, "model")
        # ============================== Placeholders ==============================
        placeholder_x = tf.placeholder(
            tf.float32, shape=([None] + params["data_shape"])
        )
        placeholder_z = tf.placeholder(tf.float32, shape=(None, params["latent_dim"]))
        placeholder_c = tf.placeholder(
            tf.float32, shape=([None] + params["data_shape"][:-1] + [1])
        )
        placeholder_suffix = tf.placeholder(tf.string)

        # Set unspecified schedule steps to default values
        for target in (config["learning_rate_schedule"], config["slope_schedule"]):
            if target["start"] is None:
                target["start"] = 0
            if target["end"] is None:
                target["end"] = config["steps"]

        # Make sure result directory exists
        # make_sure_path_exists(config['result_dir'])

        # Setup GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]

        # ================================= Model ==================================
        # Create sampler configurations

        sampler_config = {
            "result_dir": str(output_dir),
            "image_grid": (config["rows"], config["columns"]),
            "suffix": placeholder_suffix,
            "midi": config["midi"],
            "colormap": np.array(config["colormap"]).T,
            "collect_save_arrays_op": config["save_array_samples"],
            "collect_save_images_op": config["save_image_samples"],
            "collect_save_pianorolls_op": config["save_pianoroll_samples"],
        }

        # Build model
        model = Model(params)

        if params.get("is_accompaniment"):
            _ = model(
                x=placeholder_x,
                c=placeholder_c,
                z=placeholder_z,
                mode="train",
                params=params,
                config=config,
            )
            predict_nodes = model(
                c=placeholder_c,
                z=placeholder_z,
                mode="predict",
                params=params,
                config=sampler_config,
            )
        else:
            _ = model(
                x=placeholder_x,
                z=placeholder_z,
                mode="train",
                params=params,
                config=config,
            )
            predict_nodes = model(
                z=placeholder_z, mode="predict", params=params, config=sampler_config
            )

        # Get sampler op
        sampler_op = tf.group(
            [
                predict_nodes[key]
                for key in ("save_arrays_op", "save_images_op", "save_pianorolls_op")
                if key in predict_nodes
            ]
        )

        # ================================== Data ==================================
        if params.get("is_accompaniment"):
            data = load_data(config["data_source"], config["data_filename"])
            #self.data = data
        # ========================== Session Preparation ===========================

        # Get tensorflow session config
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        # Create saver to restore variables
        saver = tf.train.Saver()

        tf_config = tf_config
        config = config

        params = params
        # =========================== Tensorflow Session ===========================
        with tf.Session(config=tf_config) as sess:
            # Restore the latest checkpoint
            with open(os.path.join(config["checkpoint_dir"], "checkpoint")) as f:
                checkpoint_name = os.path.basename(f.readline().split()[1].strip('"'))
            checkpoint_path = os.path.realpath(
                os.path.join(config["checkpoint_dir"], checkpoint_name)
            )
            saver.restore(sess, checkpoint_path)

            # Run sampler op
            for i in range(config["runs"]):
                feed_dict_sampler = {
                    placeholder_z: scipy.stats.truncnorm.rvs(
                        config["lower"],
                        config["upper"],
                        size=(
                            (config["rows"] * config["columns"]),
                            params["latent_dim"],
                        ),
                    ),
                    placeholder_suffix: str(i),
                }
                if params.get("is_accompaniment"):
                    sample_x = get_samples(
                        (config["rows"] * config["columns"]),
                        data,
                        use_random_transpose=config["use_random_transpose"],
                    )
                    feed_dict_sampler[placeholder_c] = np.expand_dims(
                        sample_x[..., params["condition_track_idx"]], -1
                    )
                sess.run(sampler_op, feed_dict=feed_dict_sampler)

        m = Multitrack(str(output_path_pianoroll))
        m.write(str(output_path_midi))

        if output_type == "audio":
            command_fs = (
                "fluidsynth -ni /usr/share/sounds/sf2/FluidR3_GM.sf2 "
                + str(output_path_midi)
                + " -F "
                + str(output_path_wav)
                + " -r 44100"
            )
            os.system(command_fs)
            # fs.midi_to_audio(str(output_path_midi), str(output_path_wav))
            subprocess.check_output(
                [
                    "ffmpeg",
                    "-i",
                    str(output_path_wav),
                    "-af",
                    "silenceremove=1:0:-50dB,aformat=dblp,areverse,silenceremove=1:0:-50dB,aformat=dblp,areverse",  # strip silence
                    str(output_path_mp3),
                ],
            )
            return output_path_mp3
        elif output_type == "midi":
            return output_path_midi
        elif output_type == "image":
            return output_path_img
