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



class ThreatDataProcessing:
    def __init__(self):
        self.load_data()

    def load_data(self):
        self.data = pd.DataFrame(Recordings.AnalogInputs.fetch1())
        print(self.data)


if __name__ == "__main__":
    tdp = ThreatDataProcessing()


