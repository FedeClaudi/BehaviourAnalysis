import sys
sys.path.append('./')
import shutil
import os
print('Importing dlc takes a while')
from deeplabcut import analyze_videos
import yaml
import pandas as pd
import cv2

from Utilities.file_io.files_load_save import load_yaml

computer = "desk"
if computer == "desk":
    from database.TablesDefinitionsV4 import *


warnings.warn("\n\n\nCOMPUTER IS: {}\n\n\n".format(computer))


class SetUpTracking:
    def __init__(self, video_folder, pose_folder):
        """[For each video in video folder check if there is a corresponding pose file, if there isn't analyse it with the correct
        dlc model [info about dlc models is in database.dlcmodels]]
        
        Arguments:
            video_folder {[str]} -- [path to video folder]
            pose_folder {[str]} -- [path to pose folder]
        """

        if computer == "desk":
            self.temp_fld = 'M:\\'  # ? on main computer
        else:
            self.temp_fld = "D:\\Fede"
        self.move_video = True

        self.video_folder = video_folder
        self.pose_folder = pose_folder

        self.video_to_process = self.get_videos_to_process()
        self.process()



    def get_videos_to_process(self):
        # Get all the pose files and then returns a list of video files that don't have a corresponding pose file
        # pose_files = [f.split('.')[0][:-5] for f in os.listdir(self.pose_folder) if '.h5' in f]
        # return [f for f in os.listdir(self.video_folder) if 'tdms' not in f and f.split('.')[0] not in pose_files
        #         and os.path.getsize(os.path.join(self.video_folder, f)) > 2000]
        return load_yaml("Utilities/file_io/files_to_track.yml")
        # return [os.path.join(self.video_folder, tt) for tt in to_track]


    def process(self):
        """
            dlc analyze_video for each video file with the correct dlc model
            rename and move .h5 and .pickle file to the pose_file folder
        """
        def move_video(complete_path, move_video_path):
            print('Moving video over')
            try:
                shutil.copy(complete_path, move_video_path)
            except:
                raise FileNotFoundError('Could not move {} to {}'.format(complete_path, move_video_path))
            else:
                print('     file moved correctly')


        for i, video in enumerate(self.video_to_process):
            print('Processing video {} of {}'.format(i+1, len(self.video_to_process)))

            # Get the DLC model config path
            if 'overview' in video.lower():
                camera = 'overview'
                config_path = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\DLC_nets\\Nets\\maze_joined-Federico-2019-02-22\\config.yml"
            elif 'threat' in video.lower():
                camera = 'threat'
                config_path = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\DLC_nets\\Nets\\threat_camera-Federico-2019-05-13\\config.yml"
                
            # Move video to local HD: otherwise analysis breaks if internet connection is unstable
            complete_path = os.path.join(self.video_folder, video)
            
            if os.path.getsize(complete_path) < 2000: continue # Check that video has frames
            
            if self.move_video:
                move_video_path = os.path.join(self.temp_fld, video)
                if os.path.isfile(move_video_path):
                    # Video already there, but is it complete
                    if not os.path.getsize(move_video_path) == os.path.getsize(complete_path):
                        # Nope, move it 
                        os.remove(move_video_path)
                        move_video(complete_path, move_video_path)
                else:
                    move_video(complete_path, move_video_path)
                
                # Check that moving video worked correctly
                if not os.path.getsize(complete_path) == os.path.getsize(move_video_path): raise ValueError('Smth went wrong while moving the video')

                # store the path on the moved video so that we can delete it at the end of the processing
                self.move_video_path = move_video_path
            try:
                cap = cv2.VideoCapture(move_video_path)
                if not cap.isOpened(): raise ValueError
            except: 
                print("The video was not moved, analysing the orignal on winstore")
                cap = cv2.VideoCapture(complete_path)
                if not cap.isOpened(): 
                    raise ValueError('Original video file might be corrupted', complete_path)
                move_video_path = complete_path # copy variable to be used below

            # Run DLC analysis
            print("""
            Video:  {}
            Camera: {}
            Config: {}
            
            """.format(move_video_path, camera, config_path))
            analyze_videos(config_path, [move_video_path], gputouse=0, save_as_csv=False)
            
            # Rename and move .h5 and .pickle
            analysis_output = [f for f in os.listdir(self.temp_fld) if '.pickle' in f or '.h5' in f]
            # if len(analysis_output) != 2:
            #     raise FileNotFoundError('Incorrect number of files after analysis: ', len(analysis_output), analysis_output)

            for f in analysis_output:
                origin = os.path.join(self.temp_fld, f)
                name, ext = f.split('.')
                correct_name = f.split('Deep')[0]+'_pose'
                dest = os.path.join(self.pose_folder, correct_name+'.'+ext)
                try:
                    shutil.move(origin, dest)
                except:
                    raise FileExistsError('Could not move pose file from {} to {}'.format(origin, dest))

            # Remove video file moved to local harddrive
            # os.remove(move_video_path)

            # All done, on to the next


    def cleanup(self):
        # Remove the video we moved over to clean up disk
        os.remove(self.move_video_path)

if __name__ == "__main__":
    if computer != "desk":
        paths = load_yaml('paths_spike1.yml')
    else:
        paths = load_yaml('paths.yml')


    SetUpTracking(os.path.join(paths['raw_data_folder'], paths['raw_video_folder']),  paths['tracked_data_folder'])
















