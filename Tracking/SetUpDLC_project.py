import deeplabcut
import os
import yaml
import random

with open('../database/data_paths.yml', 'r') as f:
    paths = yaml.load(f)


# Get list of vidos to be traind
use_trial_clips = True
select_random_subset = .05  # if False take all video, if float between 0-1 select that proprtion of values
if use_trial_clips:
    dr = os.path.join(paths['raw_data_folder'], paths['trials_clips'])
    if not select_random_subset: select_random_subset = 10  # set to arbritarily high value

    training_videos = [str(os.path.join(dr, v)) for v in os.listdir(dr) if random.uniform(0.0, 1.0) < select_random_subset]

else:
    raise ValueError('Feature not yet implemetned: use other folders to select training videos')

# Create new video
os.chdir(paths['dlc_nets'])
# deeplabcut.create_new_project('Maze', 'Federico', training_videos, str('D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\DLC_nets\\Nets'), copy_videos=True)


# Extract frames
cfg_path = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\DLC_nets\\Nets\\Maze-Federico-2018-11-07\\config.yaml'
# deeplabcut.extract_frames(cfg_path, 'automatic', 'kmeans', crop=False, checkcropping=False)

# Label frames
deeplabcut.label_frames(cfg_path)











"""
useufl functions:

ADD NEW VIDEOS TO EXISTING PROJECT
deeplabcut.add_new_videos(`Full path of the project configuration file*',
[`full path of video 4', `full path of video 5'],copy_videos=True/False)


MANUALLY EXTRACT MORE FRAMES
deeplabcut.extract_frames(‘config_path’,‘manual’)

"""




