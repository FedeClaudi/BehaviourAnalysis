import sys
sys.path.append('./')

import os

import deeplabcut

from Utilities.video_editing import VideoConverter
from Utilities.file_io.files_load_save import load_yaml

class SetUpTracking:
    def __init__(self, rawfolder, config_file, processedfolder=None, given_list=None, overwrite=False):
        if processedfolder is None: processedfolder = rawfolder

        self.raw = rawfolder
        self.proc = processedfolder
        self.cfg = config_file

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
        deeplabcut.analyze_videos(self.cfg, [os.path.join(self.raw, f) for f in to_process_videos], gputouse=0, save_as_csv=False)

       
if __name__ == "__main__":
    paths = load_yaml('paths.yml')

    SetUpTracking(os.path.join(paths['raw_data_folder'], paths['raw_video_folder']), 
                  paths['dlc_config'], processedfolder= os.path.join(paths['raw_data_folder'], paths['raw_metadata_folder']))
















