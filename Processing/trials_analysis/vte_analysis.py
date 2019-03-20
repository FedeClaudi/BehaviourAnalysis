import sys
sys.path.append('./')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple
from itertools import combinations
import time
import scipy.stats as stats
import math
import matplotlib.mlab as mlab
import matplotlib as mpl
from scipy.signal import medfilt as median_filter
from sklearn.preprocessing import normalize
import seaborn as sns

mpl.rcParams['text.color'] = 'k'
mpl.rcParams['xtick.color'] = 'k'
mpl.rcParams['ytick.color'] = 'k'
mpl.rcParams['axes.labelcolor'] = 'k'

from database.NewTablesDefinitions import *

from Processing.tracking_stats.math_utils import *
from Utilities.file_io.files_load_save import load_yaml

from database.database_fetch import *




trials = (AllTrials & "is_escape='true'").fetch("session_uid", "experiment_name", "tracking_data", "escape_arm") 

arms=['Left_Far', 'Centre', 'Right2']

trial_t = namedtuple("t", "x y s")

template = get_maze_template('model based')

f, axarr = plt.subplots(ncols=3, nrows=3)
f2, ax2 = plt.subplots()
ax2.imshow(template)
tot, byarm = 0, [0, 0, 0]
for uid, experiment, tracking, escape_arm in zip(*trials):
    if not experiment in ['Model Based']: continue

    y = tracking[:, 1, :]
    if np.any(y[0] > 300): continue
    # thresh = np.where(y>400)[0][0]
    thresh = -1
    trial = trial_t(*[tracking[:thresh, i, :] for i in [0, 1, 2]])

    if np.min(trial.x)< 300: escape_arm='Left_Far'

    # theta = math.atan2(line_smoother(np.diff(trial.x)),line_smoother(np.diff(trial.y)))

    ax = axarr[0, arms.index(escape_arm)]
    ax.scatter(trial.x, trial.y,  c=trial.s, alpha=.15, s=50)

    tot += 1
    if escape_arm == 'Centre':
        c = 'r'
        byarm[0] += 1
    elif escape_arm == 'Right2':
        c = 'g'
        byarm[1] += 1
    else: 
        byarm[2] += 1

    ax2.scatter(trial.x, trial.y,  c=trial.s, alpha=1, s=50)
ax2.set(xlim=[0, 1000], ylim=[0, 1000])
p_byarm = [c/tot for c in byarm]
x = np.arange(len(arms))
colors = get_n_colors(3)
axarr[1, 0].bar(x, p_byarm, color=colors)
axarr[1, 0].set(xticks=x, xticklabels=arms)
    



plt.show()

