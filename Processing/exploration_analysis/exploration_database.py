import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from pandas.plotting import scatter_matrix
from collections import namedtuple
from itertools import combinations

from database.NewTablesDefinitions import *
from database.dj_config import start_connection

from Processing.tracking_stats.math_utils import line_smoother
from Utilities.file_io.files_load_save import load_yaml
from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame

from database.database_fetch import *


"""
    This script should collect all explorations from all sessions into a single database table

    An exploration is defined as the time between 60s after the first recording,
    starts to just before the first stimulus.

    In addition to the tracking data, in the table we should record:
        - How much did the mouse travel during exploration
        - How long did the exploration last
        - How much time the mouse spend on the shelter platform

    definition = 
        exploration_id: int
        ---
        recording_uid: varchar(128)
        experiment_name: varchar(128)
        tracking_data: longblob
        total_travel: int               # Total distance covered by the mouse
        tot_time_in_shelter: int        # Number of seconds spent in the shelter
        duration: int                   # Total duration of the exploration in seconds
        median_vel: in                  # median velocity in px/s during exploration
    

""" 



class AllExplorationsPopulate:
    def __init__(self, erase_table=False, fill_in_table=False):
        if erase_table:
            AllExplorations.drop()
            print('Table erased, exiting...')
            sys.exit()

        if fill_in_table:
            self.cutoff = 120  # ! Number of seconds to skip at the beginning of the first recording
            print('Fetching data...')

            self.populate()
    
    def populate(self):
        





    def calculations_on_tracking_data(self, data, fps):
        """[Given the tracking data for an exploration, calc median velocity, distance covered and time in shetler]
        
        Arguments:
            data {[np.array]} -- [Exploration's tracking data]
        """

        # Calc median velocity in px/s
        median_velocity = np.nanmedian(data[:, 2])*fps

        # Calc time in shelter
        time_in_shelt = int(round(np.where(data[:, -1]==0)[0].shape[0]/fps))

        # Calc time on T
        time_on_t = int(round(np.where(data[:, -1]==1)[0].shape[0]/fps))

        # Calc total distance covered
        distance_covered = int(round(np.sum(data[:, 2])))

        # Calc duration in seconds
        duration = int(round(data.shape[0]/fps))

        return median_velocity, time_in_shelt, time_on_t, distance_covered, duration







                        


if __name__ == "__main__":
    print(AllExplorations())
    AllExplorationsPopulate(erase_table=False, fill_in_table=True)

    print(AllExplorations())