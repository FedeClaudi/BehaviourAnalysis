import pandas as pd
import os

import sys
sys.path.append('./') 

from database.dj_config import start_connection
start_connection()
from database.Populate_database import PopulateDatabase
from database.NewTablesDefinitions import *
import time

from Utilities.video_and_plotting.video_editing import VideoConverter, Editor
from Utilities.file_io.sort_behaviour_files import sort_mantis_files


""" [Toolbox functions to automate handling of files (e.g. convert videos from .tdms to .avi...)]
"""

class FilesAutomationToolbox:
    def __init__(self):
        # self.database = PopulateDatabase()
        pass

    def convert_tdms_to_mp4(self):
        """
        Keeps calling video conversion tool, regardless of what happens
        """
        try:
            sort_mantis_files()
        except:
            pass

        fld = 'Z:\\branco\\Federico\\raw_behaviour\\maze\\video'
        toconvert = [f for f in os.listdir(fld) if '.tdms' in f]
        
        while True:
            try:
                for f in toconvert:
                    converter = VideoConverter(os.path.join(fld, f), extract_framesize=True)
                print('All files converted, yay!')

                editor = Editor()
                editor.concated_tdms_to_mp4_clips(fld)
                print('All clips joined, yay!')

                break
            except: # ignore exception and try again
                print('Failed again at {}, trying again..\n\n'.format(time.localtime()))

    def extract_postures(self):
        pass
        """
            Checks entries in Recordings that don't have .h5 files associated and uses DLC to extract the posture

            Then calls the auto populate method for the TrackingData table to fill these in
        """

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

