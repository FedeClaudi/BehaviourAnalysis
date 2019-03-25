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


class AllExplorationsPopulate:
    def __init__(self, erase_table=False, fill_in_table=False):
        if erase_table:
            AllExplorations.drop()
            print('Table erased, exiting...')
            sys.exit()

        if fill_in_table:
            self.cutoff = 120  # ! Number of seconds to skip at the beginning of the first recording
            print('Fetching data...')

            self.table = AllExplorations()

            self.populate()
    
    def populate(self):
        sessions, session_names, experiments = (Sessions).fetch("uid","session_name", "experiment_name")
        sessions_in_table = [int(s) for s in (AllExplorations).fetch("session_uid")]

        for n, (uid, sess_name, exp) in enumerate(sorted(zip(sessions, session_names, experiments))):
            if n in [102, 132]: continue   # ! these sessions cause problems for some reason
            print(' Processing session {} of {} - {}'.format(n, len(sessions), sess_name))

            if uid in sessions_in_table: continue

            # Get the first stim of the session
            session_stims = get_stimuli_given_sessuid(uid, as_dict=True)
            if session_stims is None:
                print('No stimuli found for session')
                continue
            else:
                first_stim = session_stims[0]

            # Get the start of the frame
            if 'stim_start' in first_stim.keys():
                start = first_stim['stim_start']
            else:
                start = first_stim['overview_frame']
            if start == -1: 
                print('No stimuli found for session')
                continue

            # Get the names of all the recordings in the session
            recordings = get_recordings_given_sessuid(uid)
            recs = [r['recording_uid'] for r in recordings]


            # Get in which recording the first stimulus happened
            first_stim_rec = recs.index(first_stim['recording_uid'])

            # Get the tracking datas
            tracking_data = {}
            useful_dims = [0, 1, 2, -1]
            bps = ['body', 'snout', 'left_ear', 'right_ear', 'neck', 'tail_base']
            for i,rec in enumerate(recs):
                if i <= first_stim_rec:
                    rec_tracking = None
                    try:
                        rec_tracking = [get_tracking_given_recuid(rec, bp=bp)[0][:, useful_dims] for bp in bps]
                        temp = np.zeros((rec_tracking[0].shape[0], rec_tracking[0].shape[1], len(rec_tracking)))

                    except:
                        body = get_tracking_given_recuid(rec, bp='body')
                        if np.any(body):
                            rec_tracking = [get_tracking_given_recuid(rec, bp=bp)[:, useful_dims] for bp in bps]
                            temp = np.zeros((rec_tracking[0].shape[0], rec_tracking[0].shape[1], len(rec_tracking)))

                    if rec_tracking is not None:
                        for tn, t in enumerate(rec_tracking):
                            temp[:, :, i] = t
                        tracking_data[rec] = temp
                    else:
                        continue

            # Crop the last tracking data to the stimulus frame
            try:
                tracking_data[first_stim['recording_uid']] = tracking_data[first_stim['recording_uid']][:start, :, :] 
            except: 
                print("skipping")
                continue

            # Get the tracking data as an array
            tracking_data_array = np.vstack([tracking_data[k] for k in sorted(tracking_data.keys())])

            # Remove the first n seconds
            fps = get_videometadata_given_recuid(first_stim['recording_uid'])
            if fps == 0: fps = 40
            cutoff = self.cutoff * fps
            tracking_data_array =tracking_data_array[cutoff:, :]

            # Get more data from the tracking
            median_velocity, time_in_shelt, time_on_t, distance_covered, duration = self.calculations_on_tracking_data(tracking_data_array, fps)


            # Get dict to insert in table
            ids_in_table = AllExplorations.fetch("exploration_id")
            expl_id = ids_in_table[-1] + 1


            key = dict(
            exploration_id = expl_id,
            session_uid= int(uid),
            experiment_name= exp,
            tracking_data = tracking_data_array,
            total_travel = int(distance_covered),
            tot_time_in_shelter = int(time_in_shelt), 
            tot_time_on_threat = int(time_on_t),
            duration = int(duration), 
            median_vel = int(median_velocity),            
            session_number_trials = len(session_stims),
            exploration_start = cutoff
            )

            self.table.insert1(key)
            

    def calculations_on_tracking_data(self, data, fps):
        """[Given the tracking data for an exploration, calc median velocity, distance covered and time in shetler]
        
        Arguments:
            data {[np.array]} -- [Exploration's tracking data]
        """

        # Calc median velocity in px/s
        median_velocity = np.nanmedian(data[:, 2])*fps

        # Calc time in shelter
        time_in_shelt = np.where(data[:, -1]==0)[0].shape[0]/fps

        # Calc time on T
        time_on_t =np.where(data[:, -1]==1)[0].shape[0]/fps

        # Calc total distance covered
        distance_covered = np.sum(data[:, 2])

        # Calc duration in seconds
        duration = data.shape[0]/fps

        return median_velocity, time_in_shelt, time_on_t, distance_covered, duration



if __name__ == "__main__":
    print(AllExplorations())
    AllExplorationsPopulate(erase_table=False, fill_in_table=True)

    print(AllExplorations())