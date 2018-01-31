import os
import yaml

with open("settings.yaml", "r") as config_file:
    settings = yaml.load(config_file)
