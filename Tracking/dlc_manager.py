# -*- coding: utf-8 -*-

print('Importing deeplabcut takes a while...')
import os
import platform
import random
import sys

import deeplabcut
import yaml
print(' ... ready!')


class DLCManager:
    
    """
    Collection of useful functions for deeplabcut:

    ADD NEW VIDEOS TO EXISTING PROJECT
    deeplabcut.add_new_videos(`Full path of the project configuration file*',
    [`full path of video 4', `full path of video 5'],copy_videos=True/False)

    MANUALLY EXTRACT MORE FRAMES
    deeplabcut.extract_frames(‘config_path’,‘manual’)
    """

    def __init__(self):
        with open('../paths.yml', 'r') as f:
            self.paths = yaml.load(f)

        with open('Tracking\dlcproject_config.yml', 'r') as f:
            self.settings = yaml.load(f)

    
    def sel_videos_in_folder(all=False, min_n=None):
        dr = self.settings['dr']
        if min_n is None: min_n = self.settings['min_num_vids']

        if all:
            videos = [os.path.join(dr,f) for f in os.listdir(dr) if 'avi' in f]
        else:  # get a subset of the videos in folder
            if len(os.listdir(dr)) < min_n:  # get as many as you can
                min_n = len(os.listdir(dr))
                self.settings['get_training_vids_th'] = 10  # high value to ensure we get all the viedeos

            while len(videos) < min_vids:
                videos = [str(os.path.join(dr, v)) for v in os.listdir(dr) 
                            if random.uniform(0.0, 1.0) < arguments['get_training_vids_th'] 
                            and 'avi' in v and str(os.path.join(dr, v)) not in videos
                            and 'labeled' in v]
        return  videos


    def create_project(self):
        # TODO add check if file existsts already
        training_videos = self.sel_videos_in_folder()
        print('Creating project with {} videos'.format(len(training_videos)))

        deeplabcut.create_new_project(arguments['project_params']['experiment'],
                                    arguments['project_params']['experimenter'], 
                                    training_videos, 
                                    working_directory=project_path, copy_videos=True)


    def add_videos_to_project(self):
        vids_to_add = self.sel_videos_in_folder()
        deeplabcut.add_new_videos(cfg_path, traivids_to_addning_videos, copy_videos=True)

    def extract_frames(self):
        deeplabcut.extract_frames(cfg_path, 'automatic', self.settings['extract_frames_mode'], crop=False, checkcropping=False)

    def label_frames(self):
        print('Getting ready to label frames')
        deeplabcut.label_frames(cfg_path)

    def check_labels(self):
        deeplabcut.check_labels(cfg_path)

    def create_training(self):
        deeplabcut.create_training_dataset(cfg_path)

    def train_network(self):
        deeplabcut.train_network(cfg_path, shuffle=1)

    def evaluate_network(self):
        deeplabcut.evaluate_network(cfg_path, plotting=True)

    def analyze_videos(self):
        deeplabcut.analyze_videos(cfg_path, test_videos, shuffle=1, save_as_csv=False)

    def create_labeled_videos(self):
        deeplabcut.create_labeled_video(cfg_path,  test_videos)

    def extract_outliers(self):
        deeplabcut.extract_outlier_frames(cfg_path, [test_videos[0]])

    def refine_labels(self):
        deeplabcut.refine_labels(cfg_path)
