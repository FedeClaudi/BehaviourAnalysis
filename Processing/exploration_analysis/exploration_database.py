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
            self.cutoff = 60  # ! Number of seconds to skip at the beginning of the first recording
            print('Fetching data...')
            self.fetch_data()

            self.populate()
    
    def fetch_data(self):
        # This table
        self.table = AllExplorations()
        self.entry_in_table = pd.DataFrame(self.table.fetch())['session_uid'].values

        # Stimuli
        self.mantis_stims = pd.DataFrame(MantisStimuli().fetch())
        self.behav_stims = pd.DataFrame(BehaviourStimuli().fetch())

        # Recordings and sessions
        self.recordings = pd.DataFrame(Recordings().fetch())
        self.sessions = pd.DataFrame(Sessions().fetch())

        # Body tracking
        fetched = pd.DataFrame((TrackingData.BodyPartData & 'bpname = "body"').fetch())
        self.tracking = fetched.loc[fetched['bpname'] == 'body']

        # VideoFiles metadata
        self.video_metadata = pd.DataFrame((VideoFiles.Metadata).fetch())



    def populate(self):
        """
            Loop over each session and find the start and stop time of the exploration.
            In particular find on which recording the first stimulus takes place.

            Once you have this get the relevant tracking data from the recordings involved

            Then calculate stuff based on the tracking data and fill in the table

        """

        # Loop over each session in the database
        for session_i, session_row in self.sessions.iterrows():
            """
                For each session get all the tracking data before the first stimulus
            """

            exp = session_row['experiment_name']
            if exp == 'Lambda Maze': continue
            session_uid = session_row['uid']

            # Avoid re processing an entry that is already in table
            if str(session_uid) in self.entry_in_table: continue

            if session_uid < 184: 
                stim_table = self.behav_stims
                backup_fps = 30
            else: 
                stim_table = self.mantis_stims  
                backup_fps = 40

            # Get all the recordings that belong to this session
            session_recs = get_recordings_given_sessuid(session_uid, self.recordings)

            # Get all stimuli that belong to this session
            session_stims = get_stimuli_given_sessuid(session_uid, stim_table)

            # Get FPS of sessions' video
            session_video_metadata = get_videometadata_given_sessuid(session_uid, self.video_metadata)
            try:
                session_fps = session_video_metadata.iloc[0]['fps']
            except:
                # Use a heuristic to determin fps
                fps = backup_fps

            # Get the cut off in frames
            exploration_cutoff = int(round(self.cutoff * session_fps))

            # Get the recording number and the frame at which the first stim took place
            try:
                first_stim  = session_stims.iloc[0]
            except:
                print('No stimuli found for session  - ', session_uid)
                continue
            try:
                fs_recnum = int(first_stim['recording_uid'].split('_')[-1])
            except:
                fs_recnum = 1 

            if 'stim_start' in first_stim.keys():
                fs_framenum = first_stim['stim_start']
            else:
                fs_framenum = first_stim['overview_frame']

            tracking_data = []
            if fs_recnum == 0:
                raise ValueError('unexpected fs recnum: ', fs_recnum)
            
            # Get the tracking data from all recordings before the one with the stim
            for reci in np.arange(start=1, stop=fs_recnum+1):
                rec = session_recs.iloc[reci-1]
                tracking = get_tracking_given_recuid(rec['recording_uid'], self.tracking)
                try:
                    tracking_data.append(tracking['tracking_data'].values[0])
                except:
                    print('no tracking data found for session - ', session_uid)
                    continue

            # If the tracking data comes from multiple sessions concatenate it
            if len(tracking_data)>1:
                # Get the last recording and crop it to stim oneset
                last_rec = tracking_data.pop()
                tracking_data.append(last_rec[:fs_framenum-1, :])

                # Stack all the tracking data
                tracking_array = np.vstack(tracking_data)

                # Remove data befofe the cutoff
                tracking_array = tracking_array[exploration_cutoff:, :]
                a = 1
            elif not tracking_data:
                continue
            else:
                tracking_array = tracking_data[0][exploration_cutoff:fs_framenum-1, :] # Take tracking data between cutoff and first stim

            """
                    CALCULATE STUFF BASED ON TRACKING DATA
            """

            median_velocity, time_in_shelt, time_on_t, distance_covered, duration = self.calculations_on_tracking_data(tracking_array,session_fps)

            """
                    ADD ENTRY TO TABLE
            """
            entry_dict = dict(
                exploration_id = session_i,
                session_uid = session_uid,
                experiment_name = exp,
                tracking_data = tracking_array,
                total_travel = distance_covered,
                tot_time_in_shelter = time_in_shelt,
                tot_time_on_threat = time_on_t, 
                duration = duration,
                median_vel = median_velocity
            )

            try:
                self.table.insert1(entry_dict)
                print('inserted: ', entry_dict['session_uid'])
            except:
                print(' !!! - did not insert !!! - ', entry_dict['session_uid'])
                if entry_dict['session_uid'] > 88:
                    raise ValueError




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