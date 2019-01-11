import pandas as pd
import os

import sys
sys.path.append('./') 

from database.dj_config import start_connection
start_connection()
from database.Populate_database import PopulateDatabase
from database.NewTablesDefinitions import *
from database.auxillary_tables import VideoTdmsMetadata

import time

from Utilities.video_and_plotting.video_editing import VideoConverter, Editor
from Utilities.file_io.sort_behaviour_files import sort_mantis_files



""" [Toolbox functions to automate handling of files (e.g. convert videos from .tdms to .avi...)]
"""

class FilesAutomationToolbox:
    def __init__(self):
        # self.database = PopulateDatabase()
        self.videos_fld = 'Z:\\branco\\Federico\\raw_behaviour\\maze\\video'
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
                raise ValueError('Could not insert: ', key)
        print(table)

    def convert_tdms_to_mp4(self):
        """
            Keeps calling video conversion tool, regardless of what happens
        """
        try:
            sort_mantis_files()
        except:
            pass

        toconvert = self.get_list_uncoverted_tdms_videos()
        while True:
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
            print("""
            Video: {}
                --- converted: {}
                --- joined: {}
            """.format(t, conv, join))
        print('To convert: ', unconverted)
        return unconverted

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
                if not number_of_frames: continue

                try:
                    if 0 in number_of_frames:  raise ValueError(number_of_frames)
                    if len(number_of_frames) not in [3, 6]: raise ValueError(number_of_frames)

                    tot = np.sum(np.array(number_of_frames))
                    # table_entry = self.video_metadata.fetch()
                    
                    metadata = pd.DataFrame(self.video_metadata.fetch())
                    entry = metadata.loc[metadata['videopath'] == os.path.join(self.videos_fld, t)]

                    if tot != entry['number_of_frames'].values: raise ValueError
                    try:
                        a = np.where(np.abs(np.diff(np.array(number_of_frames)))>1)[0][0]
                    except: 
                        print(' ... ',  number_of_frames)
                    else: 
                        
                        raise ValueError(a, number_of_frames)
                except:
                    raise ValueError('Incorrect number of frames in clips for video: ', t)

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
    # automation.convert_tdms_to_mp4()
    automation.get_list_uncoverted_tdms_videos()
    # automation.check_video_conversion_correct()
    # automation.extract_videotdms_metadata()

    # automation.macro()
