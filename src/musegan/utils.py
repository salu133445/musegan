"""This file defines some utility functions."""
import os
import errno
import shutil
import logging
import importlib
import yaml
from musegan.config import FILE_LOGLEVEL, FILE_LOG_FORMAT

# --- Path utilities -----------------------------------------------------------
def make_sure_path_exists(path):
    """Create intermidate directories if the path does not exist."""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

# --- Experiment utilities -----------------------------------------------------
def backup_src(dst):
    """Backup the source code."""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(
        os.path.dirname(os.path.realpath(__file__)), dst,
        ignore=shutil.ignore_patterns('__pycache__'))

# --- Parameter file and dictionary utilities ----------------------------------
def load_yaml(filename):
    """Load a yaml file and return as a Python object."""
    with open(filename) as f:
        return yaml.safe_load(f)

def update_not_none(dict1, dict2):
    """Update the values of keys in `dict1` with the values of the same key from
    `dict2` if the values in `dict2` is not None."""
    for key, value in dict2.items():
        if value is not None:
            dict1[key] = value

def update_existing(dict1, dict2):
    """Update the values of keys in `dict1` with the values of the same key from
    `dict2` if the values in `dict2` is not None and the same key is in `dict1`.
    """
    for key, value in dict2.items():
        if value is not None and key in dict1:
            dict1[key] = value

def load_params(params_file_path):
    """Load and return the hyperparameters."""
    # Load the default hyperparameters
    params = load_yaml(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'default_params.yaml'))

    # Load the hyperparameter file if given and update the hyperparameters
    if params_file_path is not None:
        loaded_params = load_yaml(params_file_path)
        update_not_none(params, loaded_params)

    return params

def load_component(component, name, class_name):
    """Load and return component network from file."""
    imported = importlib.import_module(
        '.'.join(('musegan.presets', component, name)))
    return getattr(imported, class_name)

# --- Logging utilities --------------------------------------------------------
def add_file_handler(logger, log_filepath, loglevel=FILE_LOGLEVEL,
                     log_format=FILE_LOG_FORMAT):
    """Add a file handler to the logger."""
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(loglevel)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

def setup_loggers(log_dir, loglevel=FILE_LOGLEVEL, log_format=FILE_LOG_FORMAT):
    """Setup the loggers with file handlers."""
    for name in logging.Logger.manager.loggerDict.keys():
        if name.startswith('musegan'):
            add_file_handler(
                logging.getLogger(name), os.path.join(log_dir, name + '.log'),
                loglevel, log_format)
