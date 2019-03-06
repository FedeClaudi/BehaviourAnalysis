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

mpl.rcParams['text.color'] = 'k'
mpl.rcParams['xtick.color'] = 'k'
mpl.rcParams['ytick.color'] = 'k'
mpl.rcParams['axes.labelcolor'] = 'k'

from database.NewTablesDefinitions import *

from Processing.tracking_stats.math_utils import *
from Utilities.file_io.files_load_save import load_yaml

from database.database_fetch import *




"""
    For each experiment, get all the explorations and look at the distribution of velocity values during the exploration.
    Then, look at the distribution of velocity of each trial divided by escape arm  and compare the two
"""



"""
    EXPLORATIONS ANALYSIS
"""
print("explorations")
def get_expl_speeds():
    exp = (AllExplorations).fetch("experiment_name", "tracking_data") 
    d = dict(
        experiment = exp[0],
        speed = [t[:, 2] for t in exp[1]]
    )
    explorations = pd.DataFrame.from_dict(d)

    experiments = sorted(set(explorations['experiment'].values))

    exploration_speeds = {}
    for experiment in experiments:
        speed = np.hstack(explorations.loc[explorations['experiment']==experiment]['speed'].values)
        exploration_speeds[experiment] =  correct_speed(speed)
    exploration_speeds['all'] = correct_speed(np.hstack(explorations['speed'].values))

    speed_th = {k: np.percentile(speed, 95) for k, speed in exploration_speeds.items()}


    return exploration_speeds, speed_th, experiments
exploration_speeds, _, experiments = get_expl_speeds()
"""
    TRIALS ANALYSIS
"""

print("trials")

def get_trials_speeds(is_escape):
    trials = (AllTrials & "is_escape='{}'".format(is_escape)).fetch("experiment_name", "tracking_data") 
    d = dict(
        experiment = trials[0],
        speed = [t[:, 2, 0] for t in trials[1]]
    )
    trials = pd.DataFrame.from_dict(d)
    trials_max_speeds = {}
    for experiment in experiments:
        speeds = list(trials.loc[trials['experiment']==experiment]['speed'].values)

        trials_max_speeds[experiment] = [np.mean(correct_speed(sp)) for sp in speeds]
    trials_max_speeds['all'] = np.hstack(list(trials_max_speeds.values()))
    return trials_max_speeds

escape_trials = get_trials_speeds("true")
not_escape_trials = get_trials_speeds("false")
all_trials = [escape_trials, not_escape_trials]

"""
    PLOTTING
"""
print("plotting")


f, axarr = plt.subplots(nrows=len(exploration_speeds), figsize=(20, 10))
colors = get_n_colors(len(exploration_speeds))

for i, (exp, speed) in enumerate(exploration_speeds.items()):
    axarr[i].hist(speed, color=colors[i], bins=100, density = True, alpha=.5)
    

    for trials, color in zip(all_trials, [colors[i], 'k']):
        trials_speeds = trials[exp]
        xx = np.random.uniform(0, 1, size=len(trials_speeds))
        axarr[i].scatter(trials_speeds, xx, c=color, s=10, alpha=.8)

    axarr[i].axvline(np.percentile(speed, 95), linestyle="--", color=colors[i] )
    print(exp, np.percentile(speed, 95))

    axarr[i].set(ylabel=exp)
    
for ax in axarr:
    ax.set(xlim=[0, 20])

plt.show()


