import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project

import matplotlib
matplotlib.use("Qt5Agg")

import os
import cv2
import datajoint as dj
import pandas as pd
import yaml
from tqdm import tqdm

from database.NewTablesDefinitions import *
from Utilities.video_and_plotting.video_editing import *
from database.database_fetch import *
from database.TablesPopulateFuncs import ToolBox


class ThreatDataProcessing:
    def __init__(self):
        self.tool_box = ToolBox()
        self.load_data()

    def load_data(self):
        rec = get_recs_given_sessuid(264)
        tdms_df, cols = self.tool_box.open_temp_tdms_as_df(rec['ai_file_path'][0], move=False, skip_df=True)
        a = 1


if __name__ == "__main__":
    tdp = ThreatDataProcessing()


