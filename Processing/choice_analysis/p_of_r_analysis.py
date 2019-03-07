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
        self.trials, self.all_trials = self.get_first_trials('true')
        self.trial_nonesc, self.all_trial_nonesc = self.get_first_trials('false')
        self.experiments = sorted(set(self.trials['experiment']))

        self.plot_pr_by_exp()
        self.plot_pr_by_mouse()
        self.plot_nonescs_vs_esc_by_exp()
        plt.show()



    def get_first_trials(self, isescape):
        # Fetch all escape trials
        to_fetch = ["origin_arm", "escape_arm", "session_uid", "recording_uid", "experiment_name", "trial_number"]
        origins, escapes, uid, rec_uid, experiment, trial_n = (AllTrials & "is_escape='{}'".format(isescape)).fetch(*to_fetch)

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

    def get_exp_pr(self, trials, exp):
        escapes = trials.loc[trials['experiment'] == exp]['escapes'].values
        return self.calc_pr(escapes)

    def get_mice_pr_given_exp(self, trials, exp):
        exp_data = trials.loc[trials['experiment'] == exp]
        sessions = set(exp_data['uid'].values)
        exp_pr = []
        for uid in sessions:
            escapes = exp_data.loc[exp_data['uid'] == uid]['escapes'].values
            exp_pr.append(self.calc_pr(escapes))
        return exp_pr


    """
            ##############################################################################################################
    """



    def plot_pr_by_exp(self):
        """
            For each experiment plot the p(R) for the first trial as a bar and for all trials as a dot
        """
        pR, pRall = {}, {}
        for exp in self.experiments:
            if ("Four" in exp or "Flip" in exp or "Close" in exp): continue
            pR[exp] = self.get_exp_pr(self.trials, exp)
            pRall[exp] = self.get_exp_pr(self.all_trials, exp)

        f, ax = plt.subplots()
        xx = np.arange(len(pR))
        yy = np.linspace(0, 1, 9)
        colors = [plt.get_cmap("tab20")(i) for i in xx]
        
        ax.bar(xx, pR.values(), color=colors, alpha=.5)
        ax.scatter(xx, pRall.values(), c=colors)
        ax.set(title="p(Right Arm)", xticks=xx, xticklabels=pR.keys(), ylabel="p", yticks=yy, ylim=[0, 1])
        ax.yaxis.grid(True)


    def plot_pr_by_mouse(self):
        pR = {}

        for exp in self.experiments:
            pR[exp] = self.get_mice_pr_given_exp(self.all_trials, exp)

        f, ax = plt.subplots()
        xx = np.arange(len(pR))
        yy = np.linspace(0, 1, 9)
        colors = [plt.get_cmap("tab20")(i) for i in xx]
        for i, (exp, pr) in enumerate(pR.items()):
            x = np.random.normal(i, .1, len(pr))
            ax.scatter(x, pr, c=colors[i], alpha=.8)
            ax.scatter(i, np.mean(pr), s=20, c='k')

        ax.set(title="p(Right Arm) by mouse", xticks=xx, xticklabels=pR.keys(), ylabel="p", yticks=yy, ylim=[-0.1, 1.1])
        ax.yaxis.grid(True)


    def plot_nonescs_vs_esc_by_mouse(self):
        pR, pRne = {}, {}

        for exp in self.experiments:
            pR[exp] = self.get_mice_pr_given_exp(self.all_trials, exp)
            pRne[exp] = self.get_mice_pr_given_exp(self.all_trial_nonesc, exp)

        f, axarr = plt.subplots(ncols=len(pR))
        xx = np.arange(len(pR))
        yy = np.linspace(0, 1, 9)
        colors = [plt.get_cmap("tab20")(i) for i in xx]
        for i, exp in enumerate(pR.keys()):
            x0 = np.zeros(len(pRne[exp]))
            x1 = np.ones(len(pR[exp]))
            axarr[i].scatter(x0, pRne[exp], c=colors[i], alpha=.75)
            axarr[i].scatter(x1, pR[exp], c=colors[i], alpha=.75)
            # axarr[0].plot(pRne[exp], pR[exp], color='k', linewidth=.75, alpha=.75)

    def plot_nonescs_vs_esc_by_exp(self):
        pR, pRne = {}, {}

        for exp in self.experiments:
            pR[exp] = self.get_exp_pr(self.all_trials, exp)
            pRne[exp] = self.get_exp_pr(self.all_trial_nonesc, exp)

        f, axarr = plt.subplots(ncols=len(pR))
        xx = np.arange(len(pR))
        yy = np.linspace(0, 1, 9)
        colors = [plt.get_cmap("tab20")(i) for i in xx]
        for i, exp in enumerate(pR.keys()):
            axarr[i].scatter(0, pRne[exp], c=colors[i], alpha=.75)
            axarr[i].scatter(1, pR[exp], c=colors[i], alpha=.75)
            axarr[i].plot([0, 1], [pRne[exp], pR[exp]], color='k', linewidth=.75, alpha=.75)
            
            if i == 0:
                axarr[i].set(title=exp, ylabel="p(R)", yticks=yy,  xticks=[0, 1], xticklabels=["Not Escape", "escape"], ylim=[0, 1], xlim=[-.1, 1.1])
            else:
                axarr[i].set(title=exp, yticks=[], xticks=[0, 1], xticklabels=["Not Escape", "escape"], ylim=[0, 1], xlim=[-.1, 1.1])




if __name__ == "__main__":
    FirstTrialAnalysis()




