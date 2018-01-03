import os
import yaml

with open("settings.yaml", "r") as config_file:
    settings = yaml.load(config_file)

settings['lmd_path'] = os.path.join(settings['lmd_dir'], 'lmd_'+settings['dataset'])
settings['lpd_path'] = os.path.join(settings['lpd_dir'], 'lpd_'+settings['dataset'])
settings['lpd_5_path'] = os.path.join(settings['lpd_dir'], 'lpd_5_'+settings['dataset'])
settings['lpd_5_cleansed'] = os.path.join(settings['lpd_dir'], 'lpd_5_cleansed')
