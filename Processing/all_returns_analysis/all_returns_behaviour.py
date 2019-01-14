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


def plot_xy(data, ax, ttl, col):
    for idx, row in data.iterrows():
        t0, t1 = row['threat_exit']-row['shelter_exit'], row['shelter_enter']-row['shelter_exit']
        tracking = row['tracking_data']
        ax.scatter(tracking[t0:t1, 0], tracking[t0:t1, 1], color=col, s=5, alpha=.05)
    ax.set(facecolor=[.2, .2 ,.2], title=ttl)

def compare_spton_vs_escape(table):
    data = pd.DataFrame(table.fetch())

    # f, axarr = plt.subplots(1, 3, facecolor=[.2, .2, .2])

    escapes = data.loc[(data['is_escape'] == 'true')&(data['arm_taken'] != 'Left_Far')&
                        (data['arm_taken'] != 'Right_Far')&(data['arm_taken']!='Centre')]

    # plot_xy(escapes, axarr[0], 'all', [.8, .8, .8])

    evoked_escapes = escapes.loc[escapes['is_trial']=='true']
    spont_escapes = escapes.loc[escapes['is_trial']=='false']

    print('Found {} evoked and {} spontaneous escapes'.format(len(evoked_escapes), len(spont_escapes)))
    # plot_xy(evoked_escapes, axarr[1], 'evoked', [.9, .6, .5])
    # plot_xy(spont_escapes, axarr[2], 'spont', [.5, .6, .9])


    r_evoked = len(evoked_escapes.loc[evoked_escapes['arm_taken'] == 'Right_Medium'])
    r_spont = len(spont_escapes.loc[spont_escapes['arm_taken'] == 'Right_Medium'])

    print("""
    {} of {} - [{}%] - evoked responses were on the right 
    {} of {} - [{}%] - spontaneous responses were on the right
    """.format(r_evoked, len(evoked_escapes), (r_evoked/len(evoked_escapes))*100,
                r_spont, len(spont_escapes), (r_spont/len(spont_escapes))*100))




def plot_dur_by_arm(table):
    data = pd.DataFrame(table.fetch())
    arms = ['Left_Far', 'Left_Medium', 'Centre', 'Right_Medium', 'Right_Far']
    f, axarr = plt.subplots(len(arms), 1)
    axarr = axarr.flatten()

    for arm, ax in zip(arms, axarr):
        sel = data.loc[(data['arm_taken'] == arm)&(data['is_trial'] == 'true')]
        ax.scatter(sel['duration'].values,np.linspace(0, sel.shape[0], sel.shape[0]), s=10, c='k', alpha=.5)
        ax.set(title=arm, xlim=[0, 100])



if __name__ == "__main__":
    plot_dur_by_arm(AllTrips())
    plt.show()