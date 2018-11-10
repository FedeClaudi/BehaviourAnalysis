print('Importing deeplabcut takes a while...')
import deeplabcut
import os
import yaml
import random

with open('../database/data_paths.yml', 'r') as f:
    paths = yaml.load(f)

arguments = dict(
    get_videos=False,
    create_proj=False,
    add_videos=False,
    extract_frames=False,
    label_frames=False,
    check_labels=False,
    create_training_set=False,
    train=False,
    evaluate=False,
    analyse_videos=True,
    create_label_video=True,
    plot_trajectories=True,
    extract_outlier=False,
    refine_labels=False
)
cfg_path = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\DLC_nets\\Nets\\Barnes-Federico-2018-11-09\\' \
           'config.yaml'

# dr = os.path.join(paths['raw_data_folder'], paths['trials_clips'])
dr = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\DAQ\\upstairs_rig\\video_clips\\videos_for_FC'
test_videos = [os.path.join(dr,f) for f in os.listdir(dr) if 'avi' in f]  # [str(os.path.join(dr, v)) for v in os.listdir(dr) if  random.uniform(0.0, 1.0) < 0.005]


# GET VIDEOS
def get_videos():
    use_trial_clips = True
    select_random_subset = .5  # if False take all video, if float between 0-1 select that proprtion of values
    if use_trial_clips:
        if not select_random_subset: select_random_subset = 10  # set to arbritarily high value

        videos = [str(os.path.join(dr, v)) for v in os.listdir(dr) if
                           random.uniform(0.0, 1.0) < select_random_subset and 'avi' in v]
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
    os.chdir(paths['dlc_nets'])
    deeplabcut.create_new_project('Barnes', 'Federico', training_videos, str('D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\DLC_nets\\Nets'), copy_videos=True)

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




