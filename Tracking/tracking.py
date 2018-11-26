import sys
sys.path.append('./')

import os
print('Importing dlc takes a while')
import deeplabcut

from Utilities.video_editing import VideoConverter
from Utilities.file_io.files_load_save import load_yaml

class SetUpTracking:
    def __init__(self, rawfolder, config_file, processedfolder=None, given_list=None, overwrite=False):
        print('Getting ready to process videos')
        if processedfolder is None: 
            processedfolder = rawfolder
            need_to_move = False  # ? Need to move .h5 and .pickle files to new folder after tracking
        else:
            need_to_move = True

        self.raw = rawfolder
        self.proc = processedfolder
        self.cfg = config_file

        to_process_videos = 1
        while to_process_videos:
            """
                When the data are on winstore, because the connection is unstable the streaming of the video can be interrupted during the analysis, 
                resulting in an error. For this reason, we keep calling the analyse function untill all the videos in the fodler have
                been process
            """
            # Get videos to process 
            if not overwrite:  # ? check which videos were already processed
                processed_videos = [f.split('Deep')[0] for f in os.listdir(self.proc) if '.h5' in f]
                to_process_videos = [f for f in os.listdir(self.raw) if f.split('.')[0] not in processed_videos
                                    and f.split('.')[-1] in ['mp4', 'avi', 'tdms']]
            else:
                to_process_videos = [f for f in os.listdir(self.raw) if f.split('.')[-1] in ['mp4', 'avi', 'tdms']]

            # Convert all to mp4
            to_convert = [f for f in to_process_videos if 'tdms' in f]
            if to_convert: raise ValueError('Feature not yet implemented: videoconversion to mp4')
            
            # Analyze
            print('Analysing {} videos with DeepLabCut'.format(len(to_process_videos)))
            try:
                deeplabcut.analyze_videos(self.cfg, [os.path.join(self.raw, f) for f in to_process_videos], gputouse=0, save_as_csv=False)
            except:
                print('\n\nSomething went wrong, starting again\n\n')

        # Move 
        if need_to_move:
            files_to_move = [f for f in os.listdir(self.raw) if '.h5' in f or '.pickle' in f]
            if len(files_to_move) != 2*len(to_process_videos):
                raise ValueError('Something went wrong when trying to move files')
            print('Moving {} files'.format(len(files_to_move)))
            
            for file in files_to_move:
                os.rename(os.path.join(self.raw, file), os.path.join(self.proc, file))

if __name__ == "__main__":
    paths = load_yaml('paths.yml')

    SetUpTracking(os.path.join(paths['raw_data_folder'], paths['raw_video_folder']), 
                  paths['dlc_config']) # , processedfolder= os.path.join(paths['raw_data_folder'], paths['raw_metadata_folder']))
















