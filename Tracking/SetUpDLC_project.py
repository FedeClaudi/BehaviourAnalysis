# -*- coding: utf-8 -*-

print('Importing deeplabcut takes a while...')
import os
import platform
import random
import sys

import deeplabcut
import yaml

with open('database/data_paths.yml', 'r') as f:
    paths = yaml.load(f)

if 'windows' in str(platform.sys).lower():
    cfg_path = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\DLC_nets\\Nets\\Barnes-Federico-2018-11-09\\' \
            'config.yaml'
    dr = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\DAQ\\upstairs_rig\\video_clips\\videos_for_FC'
    proj_path = str('D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\DLC_nets\\Nets')
else:
    cfg_path = "/Users/federicoclaudi/Desktop/testgui-Federico-2018-11-12/config.yaml"
    dr = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/raw_data/video"
    project_path = 'Users/federicoclaudi/Desktop'

arguments = dict(
    get_training_vids_th = 0.02,
    project_params = dict(
        experiment = 'testgui',
        experimenter = 'Federico',
        project_path = project_path),
    get_videos=False,
    create_proj=False,
    add_videos=False,
    extract_frames=False,
    label_frames=True,
    check_labels=False,
    create_training_set=False,
    train=False,
    evaluate=False,
    analyse_videos=False,
    create_label_video=False,
    plot_trajectories=False,
    extract_outlier=False,
    refine_labels=False
)

test_videos = [os.path.join(dr,f) for f in os.listdir(dr) if 'avi' in f]

# GET VIDEOS
def get_videos(min_vids=5):
    use_trial_clips = True
    videos = []
    if use_trial_clips:
        if len(os.listdir(dr)) < min_vids:  # get as many as you can
            min_vids = len(os.listdir(dr))
            arguments['get_training_vids_th'] = 10  # high value to ensure we get all the viedeos

        while len(videos) < min_vids:
            videos = [str(os.path.join(dr, v)) for v in os.listdir(dr) if
                            random.uniform(0.0, 1.0) < arguments['get_training_vids_th'] 
                            and 'avi' in v and str(os.path.join(dr, v)) not in videos]
        return  videos
    else:
        raise ValueError('Feature not yet implemetned: use other folders to select training videos')

if arguments['get_videos']:
    # Get list of vidos to be traind
    training_videos = get_videos()
else:
    training_videos = None

# CREATE PROJECT
if arguments['create_proj']:
    if training_videos is None:
        training_videos = get_videos()

    print('Creating project with {} videos'.format(len(training_videos)))
    # yn = input('Continue? y/n')
    # if 'y' not in yn.lower(): sys.exit()
    os.chdir('Users/federicoclaudi/Desktop')
    
    deeplabcut.create_new_project(arguments['project_params']['experiment'],
                                  arguments['project_params']['experimenter'], 
                                  training_videos, 
                                  working_directory='/Users/federicoclaudi/Desktop', copy_videos=True)

# ADD VIDEOS TO PROJECT
if arguments['add_videos']:
    if training_videos is None:
        training_videos = get_videos()
    deeplabcut.add_new_videos(cfg_path, training_videos, copy_videos=True)

# EXTRACT FRAMES
if arguments['extract_frames']:
    deeplabcut.extract_frames(cfg_path, 'automatic', 'uniform', crop=False, checkcropping=False)

# LABEL FRAMES
if arguments['label_frames']:
    deeplabcut.label_frames(cfg_path)

# CHECK LABELS
if arguments['check_labels']:
    deeplabcut.check_labels(cfg_path)

# CREATE TRAINING SET
if arguments['create_training_set']:
    deeplabcut.create_training_dataset(cfg_path)

# TRAIN NETWORK
if arguments['train']:
    deeplabcut.train_network(cfg_path, shuffle=1)

# EVALUATE NETWORK
if arguments['evaluate']:
    deeplabcut.evaluate_network(cfg_path, plotting=True)

# ANALYZE VIDEO
if arguments['analyse_videos']:
    deeplabcut.analyze_videos(cfg_path, test_videos, shuffle=1, save_as_csv=False)

# CREATE LABELED VIDEO
if arguments['create_label_video']:
    deeplabcut.create_labeled_video(cfg_path,  test_videos)

# PLOT TRAJECTORIES
if arguments['plot_trajectories']:
    deeplabcut.plot_trajectories(cfg_path, test_videos)

# EXTRACT OUTLIERS
if arguments['extract_outlier']:
    deeplabcut.extract_outlier_frames(cfg_path, [test_videos[0]])

# REFINE LABELS
if arguments['refine_labels']:
    deeplabcut.refine_labels(cfg_path)

"""
useful functions:

ADD NEW VIDEOS TO EXISTING PROJECT
deeplabcut.add_new_videos(`Full path of the project configuration file*',
[`full path of video 4', `full path of video 5'],copy_videos=True/False)


MANUALLY EXTRACT MORE FRAMES
deeplabcut.extract_frames(‘config_path’,‘manual’)

"""
