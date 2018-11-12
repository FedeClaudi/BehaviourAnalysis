# -*- coding: utf-8 -*-

print('Importing deeplabcut takes a while...')
import os
import platform
import random
import sys
import platform

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
        # Get paths and settings
        with open('paths.yml', 'r') as f:
            self.paths = yaml.load(f)

        with open('Tracking\dlcproject_config.yml', 'r') as f:
            self.settings = yaml.load(f)

        if 'windows' in platform.system().lower():
            self.dlc_paths = self.settings['paths-windows']
        else:
            self.dlc_paths = self.settings['paths-mac']

    ### MACROS

    def initialise_project(self):
        """  Create a projec with the training videos, extract the frames and start labeling gui """
        print('Creating project')
        self.create_project()
        print('Extracting frames')
        self.extract_frames()
        print('Labeling frames')
        self.label_frames()
    
    ### UTILS

    def sel_videos_in_folder(self, all=False, min_n=None):
        print('Getting videos')
        dr = self.dlc_paths['dr']
        if min_n is None: min_n = self.settings['min_num_vids']

        if all:
            videos = [os.path.join(dr,f) for f in os.listdir(dr) if 'avi' in f]
        else:  # get a subset of the videos in folder
            if len(os.listdir(dr)) < min_n:  # get as many as you can
                min_n = len(os.listdir(dr))
                self.settings['get_training_vids_th'] = 10  # high value to ensure we get all the viedeos
            videos = []
            while len(videos) < min_n:
                videos = [str(os.path.join(dr, v)) for v in os.listdir(dr) 
                            if random.uniform(0.0, 1.0) < self.settings['get_training_vids_th'] 
                            and self.settings['video_format'] in v and str(os.path.join(dr, v)) not in videos
                            and 'labeled' not in v]
                self.settings['get_training_vids_th'] += self.settings['get_training_vids_th']/2

        yn = input('\n\nGot {} videos, continue? y/n   '.format(len(videos)))
        if yn == 'y':
            print('Got videos')
            return  videos
        else:
            sys.exit()

    ### DLC functions

    def create_project(self):
        # TODO add check if file existsts already
        training_videos = self.sel_videos_in_folder()
        print('Creating project with {} videos'.format(len(training_videos)))

        deeplabcut.create_new_project(self.settings['experiment'], self.settings['experimenter'], 
                                      training_videos, working_directory=self.dlc_paths['project_path'], copy_videos=True)

    def add_videos_to_project(self):
        vids_to_add = self.sel_videos_in_folder()
        deeplabcut.add_new_videos(cfg_path, traivids_to_addning_videos, copy_videos=True)

    def extract_frames(self):
        deeplabcut.extract_frames(self.dlc_paths['cfg_path'], 'automatic', self.settings['extract_frames_mode'], crop=False, checkcropping=False)

    def label_frames(self):
        print('Getting ready to label frames')
        deeplabcut.label_frames(self.dlc_paths['cfg_path'])

    def check_labels(self):
        deeplabcut.check_labels(self.dlc_paths['cfg_path'])

    def create_training_dataset(self):
        deeplabcut.create_training_dataset(self.dlc_paths['cfg_path'])

    def train_network(self):
        deeplabcut.train_network(self.dlc_paths['cfg_path'], shuffle=1)

    def evaluate_network(self):
        deeplabcut.evaluate_network(self.dlc_paths['cfg_path'], plotting=True)

    def analyze_videos(self):
        vids = self.sel_videos_in_folder()
        deeplabcut.analyze_videos(self.dlc_paths['cfg_path'], vids, shuffle=1, save_as_csv=False)

    def create_labeled_videos(self):
        vids = self.sel_videos_in_folder()
        deeplabcut.create_labeled_video(self.dlc_paths['cfg_path'],  vids)

    def extract_outliers(self):
        vids = self.sel_videos_in_folder()
        deeplabcut.extract_outlier_frames(self.dlc_paths['cfg_path'], vids)

    def refine_labels(self):
        deeplabcut.refine_labels(self.dlc_paths['cfg_path'])


if __name__ == "__main__":
    manager = DLCManager()

    manager.label_frames()

