import pandas as pd
import os

import sys
sys.path.append('./') 

from database.dj_config import start_connection
start_connection()
from database.Populate_database import PopulateDatabase
from database.Tables_definitions import *

from Utilities.video_and_plotting.video_editing import VideoConverter



""" [Toolbox functions to automate handling of files (e.g. convert videos from .tdms to .avi...)]
"""

class FilesAutomationToolbox:
    def __init__(self):
        self.database = PopulateDatabase()

    def convert_tdms_to_mp4(self):
        """
            checks for entries in Recordings that have .tdms videofiles and don't have a converted
            videofile entry and converts it + edits the entry so that the path to the new file is stored
        """

        recordings = pd.DataFrame(Recordings.fetch())
        videos = pd.DataFrame(Recordings.VideoFiles.fetch())
        converted = pd.DataFrame(Recordings.ConvertedVideoFiles.fetch())
        
        for row in recordings.itertuples(index=True, name='Pandas'):
            rec = getattr(row, 'recording_uid')
            print('Recording name: ', rec)

            rec_vids = videos.loc[videos['recording_uid']==rec]
            rec_conv_vids = converted.loc[videos['recording_uid']==rec]
            print(rec_vids, rec_conv_vids)

            # TODO use this information to convert missing videos.. 

    def extract_postures(self):
        pass
        """
            Checks entries in Recordings that don't have .h5 files associated and uses DLC to extract the posture

            Then calls the auto populate method for the TrackingData table to fill these in
        """

if __name__ == "__main__":
    automation = FilesAutomationToolbox()
    automation.convert_tdms_to_mp4()

