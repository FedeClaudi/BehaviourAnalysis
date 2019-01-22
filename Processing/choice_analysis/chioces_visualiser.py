import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from pandas.plotting import scatter_matrix
from collections import namedtuple
from itertools import combinations
import time

from database.NewTablesDefinitions import *
from database.dj_config import start_connection

from Processing.tracking_stats.math_utils import line_smoother
from Utilities.file_io.files_load_save import load_yaml
from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame






class ChoicesVisualiser:
    def __init__(self):
        # Get variables
        self.recording_uid, self.is_trial, self.is_escape, self.experiment_name, self.arm_taken = AllTrips.fetch('recording_uid', 'is_trial', 'is_escape', 'experiment_name', 'arm_taken')
        self.recordings_sessions_lookup, self.sessions_recordings_lookup = self.get_sessions()

        data_dataframe = gaireugbraeigb # TODO create a dataframe to organise and facilitate data manipulation

        self.plot_choice_by_exp()


    def get_sessions(self):
        # Get the name of each session that a recording belongs to -    WIP 
        recs_recuid, recs_sesuid = Recordings.fetch('recording_uid', 'uid')

        # Get all recordings for each session
        sessions_recordings_lookup = {}
        for sess in set(recs_sesuid):
            sessions_recordings_lookup[sess] = [r for r,s in zip(recs_recuid, recs_sesuid) if s == sess]
        print(sessions_recordings_lookup)

        return {r:s for r,s in zip(recs_recuid, recs_sesuid)}, sessions_recordings_lookup


    def plot_choice_by_exp(self):
        for exp in set(self.experiment_name):
            escape_arms = [a for i,a in enumerate(self.arm_taken) if self.experiment_name[i]==exp and self.is_escape[i]=='true']
            arms = set(escape_arms)
            arms_probs = {arm:escape_arms.count(arm)/len(escape_arms) for arm in arms}
            y = np.arange(len(arms))

            f, ax = plt.subplots()
            ax.bar(y, arms_probs.values())
            ax.set(xticks=y, xticklabels=list(arms_probs.keys()), title=exp)

    def plot_choiche_by_exp_and_session(self):
        for session, recordings in self.sessions_recordings_lookup:
            


if __name__ == "__main__":
    ChoicesVisualiser()

    plt.show()