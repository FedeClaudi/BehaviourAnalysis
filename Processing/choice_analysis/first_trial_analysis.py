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
        self.trials, self.all_trials = self.get_first_trials()
        self.experiments = sorted(set(self.trials['experiment']))

        self.plot_pr_by_exp()



    def get_first_trials(self):
        # Fetch all escape trials
        to_fetch = ["origin_arm", "escape_arm", "session_uid", "recording_uid", "experiment_name", "trial_number"]
        origins, escapes, uid, rec_uid, experiment, trial_n = (AllTrials).fetch(*to_fetch)

        # Get  the index of the first trial of each session
        first_idxs = [list(uid).index(sess) for sess in set(uid)]

        # turn lists to df
        d = dict(
            uid = uid,
            rec_uid = rec_uid,
            experiment = experiment, 
            trial_n = trial_n, 
            origins = origins,
            escapes = escapes
        )

        df = pd.DataFrame.from_dict(d)

        # Select only first trials
        return df.iloc[first_idxs], df

    @staticmethod
    def calc_pr(escapes):
        right_escapes = [e for e in escapes if "Right" in e]
        return len(right_escapes) / len(escapes)

    def plot_pr_by_exp(self):
        
        pR, pRall = {}, {}
        for exp in self.experiments:
            if ("Four" in exp or "Flip" in exp or "Close" in exp): continue
            escapes = self.trials.loc[self.trials['experiment'] == exp]['escapes'].values
            all_escapes = self.all_trials.loc[self.all_trials['experiment'] == exp]['escapes'].values
            pR[exp] = self.calc_pr(escapes)
            pRall[exp] = self.calc_pr(all_escapes)

        f, ax = plt.subplots()
        xx = np.arange(len(pR))
        yy = np.linspace(0, 1, 9)
        colors = [plt.get_cmap("tab20")(i) for i in xx]
        
        ax.bar(xx, pR.values(), color=colors, alpha=.5)
        ax.scatter(xx, pRall.values(), c=colors)
        ax.set(title="p(Right Arm)", xticks=xx, xticklabels=pR.keys(), ylabel="p", yticks=yy, ylim=[0, 1])
        ax.yaxis.grid(True)
        plt.show()







if __name__ == "__main__":
    FirstTrialAnalysis()




