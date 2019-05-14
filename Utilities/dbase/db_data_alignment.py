import sys
sys.path.append('./')  

import matplotlib
matplotlib.use("Qt5Agg")

import os
import cv2
import pandas as pd
import yaml
from tqdm import tqdm

if sys.platform != "darwin":
    import datajoint as dj
    from database.NewTablesDefinitions import *
    from database.database_fetch import *

from database.TablesPopulateFuncs import ToolBox

from Utilities.video_and_plotting.video_editing import *
from Utilities.file_io.files_load_save import load_feather
from Utilities.Maths.filtering import butter_lowpass_filter
from Processing.tracking_stats.math_utils import find_peaks_in_signal

class ThreatDataProcessing:
    def __init__(self, test_mode=False):
        self.tool_box = ToolBox()

        self.overview_ch = "/'OverviewCameraTrigger_AI'/'0'"
        self.threat_ch = "/'ThreatCameraTrigger_AI'/'0'"
        self.frame_times = {}


        if test_mode:
            self.test_folder = "/Volumes/FEDE'S STUF/TEST FILES"
            self.test_file = "190426_CA532_1.ft"
            self.start_time, self.end_time = int(1.8*10**7), int(2*10**7)  # cut the dataframe to speed up stuff


            self.make_feathers()
            self.load_a_feather()
    
    def make_feathers(self):
        for f in os.listdir(self.test_folder)[::-1]:
            if ".tdms" in f:
                feather_name = f.split(".")[0]+".ft"
                if not feather_name in os.listdir(self.test_folder):
                    content = self.tool_box.open_temp_tdms_as_df(os.path.join(self.test_folder, f), move=False, skip_df=False, memmap_dir=self.test_folder )
                    print("         ... saving")
                    content[0].to_feather(os.path.join(self.test_folder, feather_name))
                    print("                ... saved")

    def load_a_feather(self):
        print("loading")
        self.data = load_feather(os.path.join(self.test_folder, self.test_file))[self.start_time: self.end_time]



    def process_channel(self, ch, key):
        # ? We need to filter because sometimes there is quite a lot of high freq noise
        # and this gets picked up as a frame otherwise
        filtered_signal = butter_lowpass_filter(self.data[ch].values, 5000, 25000)
        self.frame_times[key] = np.add(find_peaks_in_signal(filtered_signal, 6, 4), self.start_time)


    def test_filter(self):
        f, ax = plt.subplots()
        filtered1 = butter_lowpass_filter(self.data[self.threat_ch], 6000, 25000)
        filtered2 = butter_lowpass_filter(self.data[self.threat_ch], 10000, 25000)

        ax.plot(self.data[self.threat_ch].values, color='k', linewidth=3, alpha=1)
        ax.plot(filtered1, color='r', linewidth=2, alpha=.5)
        ax.plot(filtered2, color='g', linewidth=2, alpha=.5)

    def plot_channels(self):
        f, ax = plt.subplots()
        ax.plot(self.data[self.overview_ch], color='c', linewidth=3)
        ax.plot(self.data[self.threat_ch], color='r', linewidth=1)

        ax.plot(self.frame_times["threat"], [-.4 for x in self.frame_times["threat"]], "o", color="r")
        ax.plot(self.frame_times["overview"], [-.8 for x in self.frame_times["overview"]], "o", color="c")

if __name__ == "__main__":
    tdp = ThreatDataProcessing(test_mode = True)

    tdp.process_channel(tdp.threat_ch, "threat")
    tdp.process_channel(tdp.overview_ch, "overview")

    tdp.plot_channels()

    # tdp.test_filter()

    plt.show()



