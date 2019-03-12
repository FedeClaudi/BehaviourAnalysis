import pandas as pd
import os
import yaml
import sys
sys.path.append('./') 

try:
    from database.Populate_database import PopulateDatabase
    from database.NewTablesDefinitions import *
    from database.auxillary_tables import VideoTdmsMetadata
except:
    pass

import time

from Utilities.video_and_plotting.video_editing import VideoConverter, Editor
from Utilities.file_io.sort_behaviour_files import sort_mantis_files



""" [Toolbox functions to automate handling of files (e.g. convert videos from .tdms to .avi...)]
"""

class FilesAutomationToolbox:
    def __init__(self):
        # self.database = PopulateDatabase()
        self.videos_fld = 'Z:\\branco\\Federico\\raw_behaviour\\maze\\video'
        self.pose_fld = 'Z:\\branco\\Federico\\raw_behaviour\\maze\\pose'
        self.video_metadata = VideoTdmsMetadata

    def macro(self):
        # Fill in metadata dj table
        try:
            self.extract_videotdms_metadata()
        except:
            pass

        # IF something needs conversion, conert it
        to_conv = self.get_list_uncoverted_tdms_videos()
        print('Converting: ', to_conv)
        if to_conv:
            self.convert_tdms_to_mp4()
        
        # Check if there was something wrong with conversion
        self.check_video_conversion_correct()

        # Join Clips
        Editor.concated_tdms_to_mp4_clips(self.videos_fld)

    def extract_videotdms_metadata(self):
        """[Populate a dj table with the videos metadata]
        """

        table = VideoTdmsMetadata()
        tdmss = [f for f in os.listdir(self.videos_fld) if '.tdms' in f]
        for t in tdmss:
            props, number_of_frames = VideoConverter.extract_framesize_from_metadata(os.path.join(self.videos_fld, t))
            key = dict(videopath=os.path.join(self.videos_fld, t),
                        width=props['width'],
                        height=props['height'],
                        number_of_frames=number_of_frames,
                        fps=props['fps'])
            try:
                table.insert1(key)
            except:
                pass
                # raise ValueError('Could not insert: ', key)
        print(table)

    def convert_tdms_to_mp4(self, n_processes=1):
        """
            Keeps calling video conversion tool, regardless of what happens
        """
        try:
            sort_mantis_files()
        except:
            pass

        toconvert = self.get_list_uncoverted_tdms_videos()
        while True:
            if n_processes> 1:
                in_process = None
                try:
                    for f in toconvert:
                        in_process = f
                        VideoConverter(os.path.join(self.videos_fld, f), extract_framesize=True)
                    print('All files converted, yay!')

                    editor = Editor()
                    editor.concated_tdms_to_mp4_clips(fld)
                    print('All clips joined, yay!')

                    break
                except: # ignore exception and try again
                    if in_process is not None:
                        pass # ! clear unfinished business 
                    print('Failed again, trying again..\n\n')
            else:
                for f in toconvert:
                    try:
                        VideoConverter(os.path.join(self.videos_fld, f), extract_framesize=True)
                    except:
                        continue

    @staticmethod
    def check_if_file_converted(name, folder):
        conv, join = False, False
        mp4s = [v for v in os.listdir(folder) if name in v and '.mp4' in v]
        # print(name)
        if mp4s:
            joined = [v for v in mp4s if 'joined' in mp4s]
            conv = True
            if joined: join=True

        return conv, join

    def get_list_uncoverted_tdms_videos(self):
        """
            Check which videos still need to be converted
        """
        tdmss = [f for f in os.listdir(self.videos_fld) if '.tdms' in f]
        unconverted = []
        for t in tdmss:
            name = t.split('.')[0]
            conv, join = self.check_if_file_converted(name, self.videos_fld)
            if not conv: unconverted.append(t)
            
        # store names to file
        store = "Utilities/file_io/files_to_convert.yml"
        with open(store, 'w') as out:
            yaml.dump(unconverted, out)


        print('To convert: ', unconverted)
        print(len(unconverted), ' files yet to convert')
        return unconverted

    def get_list_not_tracked_videos(self):
        videos = [f.split('.')[0] for f in os.listdir(self.videos_fld) if 'tdms' not in f]
        poses = [f.split('_')[:-1] for f in os.listdir(self.pose_fld) if 'h5' in f]

        not_tracked = [f for f in videos if f.split('_') not in poses and 'overview'  in f.lower()]
        print('To track: ', not_tracked)
        print(len(not_tracked), ' files yet to track')
        return not_tracked

    def check_video_conversion_correct(self):
        """[Check if the video conversion was done correctly, i.e. we have all the frames and all clips have same length]
        
        Raises:
            ValueError -- [If something went wrong raise an error]
        """

        tdmss = [f for f in os.listdir(self.videos_fld) if '.tdms' in f]
        editor = Editor()

        for t in tdmss:
            print('\n\nChecking: ', t)
            name = t.split('.')[0]
            conv, join = self.check_if_file_converted(name, self.videos_fld)
            number_of_frames = []
            if conv and not join:
                mp4s = [v for v in os.listdir(self.videos_fld) if name in v and '.mp4' in v]
                for mp4 in mp4s:
                    if [f for f in ['top', 'side', 'catwalk'] if f in mp4]: continue # ignore cropped videos
                    cap = cv2.VideoCapture(os.path.join(self.videos_fld, mp4))
                    nframes, width, height, fps = editor.get_video_params(cap)
                    number_of_frames.append(nframes)
                if not number_of_frames or number_of_frames[0] == 0:
                    continue 

                # Check that we are sure about the total number of frames
                tot = np.sum(np.array(number_of_frames))
                metadata = pd.DataFrame(self.video_metadata.fetch())
                entry = metadata.loc[metadata['videopath'] == os.path.join(self.videos_fld, t)]
                if not entry.shape[0]: raise FileNotFoundError('Load video metadata first')

                if tot != entry['number_of_frames'].values[0]: raise ValueError('Tot frames, converted frames:', a, number_of_frames)

                else: 
                    print(' ...  correctly converted all ',  tot, 'frames [{} converted]'.format(tot))
                # else: 
                #     raise ValueError('Tot frames, converted frames:', a, number_of_frames)

    def remove_stupid_videofiles(self):
        dr = 'Z:\\branco\\Federico\\raw_behaviour\\maze\\video'
        for f in os.listdir(dr):
            if 'top' in f or 'side' in f or 'catwalk' in f:
                if os.path.getsize(os.path.join(dr, f)) < 2000:
                    os.remove(os.path.join(dr, f))
            


    def fillin_incompletevideos(self):
        inc_table = VideosIncomplete()
        rec_table = Recordings()
        vid_table = VideoFiles()
        recordings = rec_table.fetch(as_dict=True)
        incompletes = inc_table.fetch(as_dict=True)
        vidoes = vid_table.fetch(as_dict=True)


        for entry in incompletes:
            print('Filling in: ', entry['uid'], entry['recording_uid'])

            if entry['conversion_needed'] == true:
                video = [v for v in videos if v['recording_uid']==entry['recording_uid']][0]
                

            if entry['dlc_needed']:
                warnings.warn('Feature not yet implemented')



if __name__ == "__main__":
    automation = FilesAutomationToolbox()

    automation.convert_tdms_to_mp4()

    automation.get_list_uncoverted_tdms_videos()
    # automation.get_list_not_tracked_videos()

    # automation.extract_videotdms_metadata()
    # automation.check_video_conversion_correct()

    # automation.macro()

    # automation.remove_stupid_videofiles()
