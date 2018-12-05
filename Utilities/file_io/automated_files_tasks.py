
import sys
sys.path.append('./') 

from database.dj_config import start_connection
start_connection()
from database.Populate_database import PopulateDatabase

from Utilities.video_and_plotting.video_editing import VideoConverter

""" [Toolbox functions to automate handling of files (e.g. convert videos from .tdms to .avi...)]
"""

class FilesAutomationToolbx:
    def __init__(self):
        self.database = PopulateDatabase()

    def convert_tdms_to_mp4(self):
        pass
        """
            checks for entries in Recordings that have .tdms videofiles and don't have a converted
            videofile entry and converts it + edits the entry so that the path to the new file is stored
        """

    def extract_postures(self):
        pass
        """
            Checks entries in Recordings that don't have .h5 files associated and uses DLC to extract the posture

            Then calls the auto populate method for the TrackingData table to fill these in
        """



