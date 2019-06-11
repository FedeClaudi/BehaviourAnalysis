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
from scipy.signal import medfilt as median_filter
import seaborn as sns

mpl.rcParams['text.color'] = 'k'
mpl.rcParams['xtick.color'] = 'k'
mpl.rcParams['ytick.color'] = 'k'
mpl.rcParams['axes.labelcolor'] = 'k'

from database.TablesDefinitionsV4 import *

from Utilities.maths.math_utils import *
from Utilities.file_io.files_load_save import load_yaml

from database.database_fetch import *




"""
    For each experiment, get all the explorations and look at the distribution of velocity values during the exploration.
    Then, look at the distribution of velocity of each trial divided by escape arm  and compare the two
"""



"""
    EXPLORATIONS ANALYSIS
"""
def get_expl_speeds():
    exp = (AllExplorations).fetch("experiment_name", "tracking_data", "session_uid") 
    d = dict(
        experiment = exp[0],
        speed = [t[:, 2] for t in exp[1]],
        uid  = exp[2]
    )
    explorations = pd.DataFrame.from_dict(d)

    experiments = sorted(set(explorations['experiment'].values))

    exploration_speeds = {}
    fpss = {}
    for experiment in experiments:
        speed = np.hstack(explorations.loc[explorations['experiment']==experiment]['speed'].values)

        one_session_for_the_exp = explorations.loc[explorations['experiment']==experiment]['uid'].values[0]
        # rec_for_that_sess = get_recs_given_sessuid(one_session_for_the_exp)['recording_uid'].values[0]

        if one_session_for_the_exp > 184: fps = 40
        else: fps = 30 # ! hardocded
        # fps = get_videometadata_given_recuid(rec_for_that_sess)

        exploration_speeds[experiment] =  correct_speed(speed)
        fpss[experiment] = fps
    exploration_speeds['all'] = correct_speed(np.hstack(explorations['speed'].values))
    fpss['all'] = 30

    speed_th = {k: np.percentile(speed, 95) for k, speed in exploration_speeds.items()}
    return exploration_speeds, speed_th, experiments, fpss


"""
    TRIALS ANALYSIS
"""
def get_trials_speeds(is_escape, fpss):
    trials = (AllTrials & "is_escape='{}'".format(is_escape)).fetch("experiment_name", "tracking_data", "escape_arm") 
    d = dict(
        experiment = trials[0],
        speed = [t[:, 2, 0] for t in trials[1]],
        escape_arm = trials[2],
    )
    trials = pd.DataFrame.from_dict(d)
    trials_max_speeds, trials_arms = {}, {}
    for experiment in experiments:
        fps = fpss[experiment]
        speeds = list(trials.loc[trials['experiment']==experiment]['speed'].values * fps) 

        trials_max_speeds[experiment] = [np.mean(correct_speed(sp)) for sp in speeds]
        trials_arms[experiment] = list(trials.loc[trials['experiment']==experiment]['escape_arm'].values)

    trials_max_speeds['all'] = np.hstack(list(trials_max_speeds.values()))
    trials_arms['all'] = trials['escape_arm'].values
    return trials_max_speeds, trials_arms


if __name__ == "__main__":
    exploration_speeds, _, experiments, fpss = get_expl_speeds()
    escape_trials, trials_arms = get_trials_speeds("true", fpss)
    not_escape_trials, not_escape_arms = get_trials_speeds("false", fpss)
    all_trials = [escape_trials, not_escape_trials]
    all_arms = [trials_arms, not_escape_arms]

    """
        PLOTTING
    """
    print("plotting")


    f, axarr = plt.subplots(nrows=len(exploration_speeds), figsize=(20, 10))
    colors = get_n_colors(len(exploration_speeds))

    for i, (exp, speed) in enumerate(exploration_speeds.items()):
        fps = fpss[exp]
        speed *= fps

        axarr[i].hist(speed, color=colors[i], bins=100, density = False, alpha=.5)

        for trials, color in zip(all_trials, [colors[i], 'k']):
            trials_speeds = trials[exp]
            xx = np.random.uniform(0, 5000, size=len(trials_speeds))
            axarr[i].scatter(trials_speeds, xx, c=color, s=10, alpha=.8)

        axarr[i].axvline(np.percentile(speed, 95), linestyle="--", color=colors[i] )
        axarr[i].set(ylabel=exp)
        
    for ax in axarr:
        ax.set(xlim=[0, 1000])


    # Plot velocity by arm
    arms_names = ['Left2', 'Left_Far', 'Left_Medium', 'Centre', 'Right_Medium', 'Right_Far', 'Right2']
    xx = np.arange(len(arms_names))
    

    for exp_n, exp in enumerate(escape_trials.keys()):
        f, ax = plt.subplots(figsize=(20, 10))

        fps = fpss[exp]
        expl_speed = exploration_speeds[exp] 
        threshold = np.percentile(expl_speed, 95)

        for trials, tarms, color in zip(all_trials, all_arms, [colors[exp_n], 'k']):
            trials_speeds = trials[exp]
            trials_speeds = [t*fps for t in trials_speeds]
            for arm_n, arm in enumerate(arms_names):
                arms_idxs = [i for i,a in enumerate(tarms[exp]) if a == arm]
                speed_by_arm = ([trials_speeds[i] for i in arms_idxs])
                x = np.random.normal(arm_n, .1, len(speed_by_arm))

                ax.scatter(x, speed_by_arm, c=color)
                ax.axvline(arm_n, linestyle=":", color="k")

                if color == "k":
                    ax.scatter(arm_n, np.mean(speed_by_arm), c='b', s=30)
                else:
                    ax.scatter(arm_n, np.mean(speed_by_arm), c='r', s=30, label="{}-{}".format(arm, round(np.mean(speed_by_arm))))
            ax.axhline(threshold, linestyle=":", color="k")
        ax.set(title=exp, ylabel="px/s", xticks=xx, xticklabels=arms_names)



    escapes = []
    escapes.extend(list(escape_trials['all']))
    escapes.extend(list(not_escape_trials['all']))
    arms = []
    arms.extend(list(trials_arms['all']))
    arms.extend(list(not_escape_arms['all']))

    f, ax = plt.subplots()
    for arm in arms_names:
        if "2" in arm: continue 
        idxs = [i for i, a in enumerate(arms) if a == arm]
        arm_escapes = [escapes[i] for i in idxs]
        sns.kdeplot(arm_escapes, shade=True, label=arm, ax=ax)
        # ax.hist(arm_escapes, bins=100, histtype ="step", label=arm, density=True)
    ax.legend()
    ax.set(title="distribution of mean spped for all trials by arm taken", ylabel="density", xlabel="px/s")


    plt.show()


