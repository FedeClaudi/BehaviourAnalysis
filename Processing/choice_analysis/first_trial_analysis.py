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
import scipy.stats as stats
import math
import matplotlib.mlab as mlab
import matplotlib as mpl

mpl.rcParams['text.color'] = 'k'
mpl.rcParams['xtick.color'] = 'k'
mpl.rcParams['ytick.color'] = 'k'
mpl.rcParams['axes.labelcolor'] = 'k'

from database.NewTablesDefinitions import *

from Processing.tracking_stats.math_utils import line_smoother
from Utilities.file_io.files_load_save import load_yaml

from database.database_fetch import *



"""
    Analyse a bunch of things related to the first stimulus evoked escape of each experiment
        - p(R)
        - p(escape) == p(origin)
        - p(R) == overall p(R)
"""


class FirstTrialAnalysis:
    def __init__(self):
        self.data, self.all_escapes = self.get_first_trials()
        print(self.data)

        self.probArm_byexp, self.arms, self.experiments = self.plt_pR_byexp()


    def get_first_trials(self):
        # Get all stim evoked escapes
        recordings, escape, origin, duration, max_speed, experiment_name = (AllTrips & 'is_escape="true"' & 'is_trial="true"').fetch('recording_uid', 'escape_arm', 'origin_arm', 'duration', 'max_speed', 'experiment_name')
        temp_d = dict(recording_uid=recordings,
                        escape_arm = escape,
                        origin_arm = origin, 
                        duration = duration,
                        max_speed = max_speed,
                        experiment_name = experiment_name)
        df = pd.DataFrame.from_dict(temp_d)

        # Get the indexes of the first escape of each session
        all_sessions = pd.DataFrame(Sessions.fetch())
        sessions = [get_sessuid_given_recuid(r, all_sessions)['session_name'].values[0] for r in set(recordings)]
        uids = {r:False for r in sessions}
        good_idxs = []
        for idx, row in df.iterrows():
            sess = get_sessuid_given_recuid(row['recording_uid'], all_sessions)['session_name'].values[0]
            if uids[sess]: continue # Make sure we keep it if it is only the first one for each session
            else:  
                uids[sess] = True
                good_idxs.append(idx)

            
        # return the filtered dataframe
        return df.iloc[good_idxs], df

    def get_probs(data, arms):
            n_escapes = data.shape[0]
            return [round(list(data['escape_arm']).count(arm)/n_escapes,6) for arm in arms]

    def plt_pR_byexp(self):
        experiments = set(self.data['experiment_name'].values)
        arms = ['Left_Far', 'Left_Medium', 'Centre', 'Right_Medium', 'Right_Far']
        y = np.arange(len(arms))
        f, axarr = plt.subplots(3, 2)
        axarr= axarr.flatten()

        probArm_by_exp = {}
        for i, exp in enumerate(sorted(experiments)):
            if 'FlipFlop' in exp: continue
            exp_data = self.data.loc[self.data['experiment_name']==exp]
            exp_data_all = self.all_escapes.loc[self.all_escapes['experiment_name']==exp]

            

            escapes_by_arm = self.get_probs(exp_data, arms)
            probArm_by_exp[exp] = escapes_by_arm
            overall_pR = get_probs(exp_data_all, ['Right_Medium'])[0]
            
            print(overall_pR)
            axarr[i].bar(y, escapes_by_arm, color='k')
            axarr[i].axhline(overall_pR, color='r', linewidth=2)
            axarr[i].set(title=exp, xticks=y,  xticklabels=list(arms))

        return probArm_byexp, arms, experiments

    def pR_firststim_vs_overall_by_mouse_by_exp(self):
        for i,exp in enumerate(sorted(self.experiments)):
            exp_data = self.all_escapes.loc[self]







if __name__ == "__main__":
    FirstTrialAnalysis()




