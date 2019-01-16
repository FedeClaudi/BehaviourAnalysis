import sys
sys.path.append('./')
import shutil
import os
print('Importing dlc takes a while')
from deeplabcut import analyze_videos

from Utilities.file_io.files_load_save import load_yaml

from database.NewTablesDefinitions import *


# class SetUpTracking:
#     def __init__(self, rawfolder, config_file, processedfolder=None, given_list=None, overwrite=False):
#         print('Getting ready to process videos')
#         if processedfolder is None: 
#             processedfolder = rawfolder
#             need_to_move = False  # ? Need to move .h5 and .pickle files to new folder after tracking
#         else:
#             need_to_move = True

#         self.raw = rawfolder
#         self.proc = processedfolder
#         self.cfg = config_file

#         to_process_videos = 1
#         while True:
#             """
#                 When the data are on winstore, because the connection is unstable the streaming of the video can be interrupted during the analysis, 
#                 resulting in an error. For this reason, we keep calling the analyse function untill all the videos in the fodler have
#                 been process
#             """
#             # Get videos to process 
#             if not overwrite:  # ? check which videos were already processed
#                 processed_videos = [f.split('Deep')[0] for f in os.listdir(self.proc) if '.h5' in f]
#                 to_process_videos = [f for f in os.listdir(self.raw) if f.split('.')[0] not in processed_videos
#                                     and f.split('.')[-1] in ['mp4', 'avi', 'tdms']]
#             else:
#                 to_process_videos = [f for f in os.listdir(self.raw) if f.split('.')[-1] in ['mp4', 'avi', 'tdms']]

#             # Convert all to mp4
#             to_convert = [f for f in to_process_videos if 'tdms' in f]
#             if to_convert: raise ValueError('Feature not yet implemented: videoconversion to mp4')
            
#             # Analyze
#             print('Analysing {} videos with DeepLabCut'.format(len(to_process_videos)))
#             try:
#                 analyze_videos(self.cfg, [os.path.join(self.raw, f) for f in to_process_videos], gputouse=0, save_as_csv=False)
#             except:
#                 print('\n\nSomething went wrong, starting again\n\n')

#         # Move 
#         if need_to_move:
#             files_to_move = [f for f in os.listdir(self.raw) if '.h5' in f or '.pickle' in f]
#             if len(files_to_move) != 2*len(to_process_videos):
#                 raise ValueError('Something went wrong when trying to move files')
#             print('Moving {} files'.format(len(files_to_move)))
            
#             for file in files_to_move:
#                 os.rename(os.path.join(self.raw, file), os.path.join(self.proc, file))


class SetUpTracking:
    def __init__(video_folder, pose_folder):
        """[For each video in video folder check if there is a corresponding pose file, if there isn't analyse it with the correct
        dlc model [info about dlc models is in database.dlcmodels]]
        
        Arguments:
            video_folder {[str]} -- [path to video folder]
            pose_folder {[str]} -- [path to pose folder]
        """

        self.video_folder = video_folder
        self.pose_folder = pose_folder

        self.dlc_models = self.get_dlc_models()
        self.video_to_process = self.get_videos_to_process()

        self.process()


    def get_dlc_models():
        return pd.DataFrame(DLCmodels().fetch())

    def get_videos_to_process():
        # Get all the pose files and then returns a list of video files that don't have a corresponding pose file
        pose_files = [f.split('.')[0] for f in os.listdir(self.pose_folder) if '.h5' in f]
        return [f for f in os.listdir(self.video_folder) if 'tdms' not in f and f.split('.')[0] not in pose_files]


    def process():
        """
            dlc analyze_video for each video file with the correct dlc model
            rename and move .h5 and .pickle file to the pose_file folder
        """

        for i, video in enumerate(self.video_to_process()):
            print('Processing video {} of {}'.format(i+1, len(self.video_to_process)))

            # Get the DLC model config path
            if 'overview' in video.lower():
                camera = 'overview'
            elif 'threat' in video.lower():
                camera = 'threat'
            else:
                camera = 'overview'  # <- behaviour software videos 

            print('     video: {}\n     camera: {}'.format(video, camer))
            config_path = self.dlc_models.loc[self.dlc_models['camera'] == camera]['cfg_file_path'].values

            # Move video to local HD: otherwise analysis breaks if internet connection is unstable
            complete_path = os.path.join(self.video_folder, video)
            move_video_path = os.path.join('M:', video)

            try:
                shutil.copy(complete_path, move_video_path)
            except:
                raise FileNotFoundError('Could not move {} to {}'.format(complete_path, move_video_path))
            else:
                print('     file moved correctly')

            # Run DLC analysis
            try:
                analyze_videos(config_path, [move_video_path], gputouse=0, save_as_csv=False)
            except:
                print('\n\nSomething went wrong, starting again\n\n')

            # Rename and move .h5 and .pickle
            analysis_output = [f for f in os.listdir('M') if '.pickle' in f or '.h5' in f]
            if len(analysis_output) != 2:
                raise FileNotFoundError('Incorrect number of files after analysis: ', len(analysis_output), analysis_output)

            for f in analysis_output:
                origin = os.path.join('M', f)
                name, ext = f.split('.')
                correct_name = f.split('Deep')[0]
                dest = os.path.join(self.pose_folder, correct_name+'.'+ext)
                try:
                    shutil.move(origin, dest)
                except:
                    raise FileExistsError('Could not move pose file from {} to {}'.format(origin, dest))

            # Remove video file moved to local harddrive
            os.remove(move_video_path)

            # All done, on to the next


if __name__ == "__main__":
    paths = load_yaml('paths.yml')

    SetUpTracking(os.path.join(paths['raw_data_folder'], paths['raw_video_folder']),
                    paths['tracked_data_folder'])
















