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


class AllExplorationsPopulate:
    def __init__(self, erase_table=False, fill_in_table=False):
        if erase_table:
            AllExplorations.drop()
            print('Table erased, exiting...')
            sys.exit()

        if fill_in_table:
            print('Fetching data...')
            self.table = AllExplorations()
            self.mantis_stims = pd.DataFrame(MantisStimuli().fetch())
            self.behav_stims = pd.DataFrame(BehaviourStimuli().fetch())
            self.tracking_data = pd.DataFrame(TrackingData().fetch())
            self.recordings = pd.DataFrame(Recordings().fetch())
            self.sessions = pd.DataFrame(Sessions().fetch())

            self.bp_tracking = pd.DataFrame(TrackingData.BodyPartData().fetch())

            self.populate()
        

    def populate(self):
        in_table = list(self.table.fetch('recording_uid'))
        for index, row in self.tracking_data.iterrows():
            # Ge the recording, session, experiment...
            recording_uid = row['recording_uid']
            if recording_uid in in_table: continue
                
            if len(recording_uid.split('_'))>2:  # if its == 2 then its the first recording
                if int(recording_uid.split('_')[-1]) != 1: continue # Only look at the first recording

            print('Processing recording:  ', recording_uid)

            recording = self.recordings.loc[self.recordings['recording_uid'] == recording_uid]
            session = self.sessions.loc[self.sessions['uid'] == recording['uid'].values[0]]
            experiment = session['experiment_name'].values[0]

            # Get the time of the first stimulus
            software = recording['software'].values[0]
            if software == 'behaviour':
                stims = self.behav_stims.loc[self.behav_stims['recording_uid'] == recording_uid]
                key = 'stim_start'
            else:
                stims = self.mantis_stims.loc[self.mantis_stims['recording_uid'] == recording_uid]
                key = 'overview_frame'

            first_stim = None
            for i, r in stims.iterrows(): # Get the time of the first stim
                if int(r['stimulus_uid'].split('_')[-1]) == 0:
                    first_stim = r[key]
            if first_stim is None: continue

            # Get the tracking data
            try:
                body_tracking = self.bp_tracking.loc[(self.bp_tracking['recording_uid']==recording_uid)&(self.bp_tracking['bpname']=='body')]['tracking_data'].values[0]
            except: pass
            if first_stim < body_tracking.shape[0]:
                tracking_data = body_tracking[:first_stim, :]
            else:
                tracking_data = body_tracking
            # except: raise ValueError

            insert_dict = dict(
                exploration_id = index, 
                recording_uid= recording_uid,
                experiment_name= experiment, 
                tracking_data = tracking_data
            )
            try:
                self.table.insert1(insert_dict)
                print('     .... inserted')
            except:
                print('     .... could not insert')

            
                        


if __name__ == "__main__":
    AllExplorationsPopulate(erase_table=False, fill_in_table=True)