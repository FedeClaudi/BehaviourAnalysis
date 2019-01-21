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
    def plotter(ax, data, color, symb, label):
        ax.plot(data['duration'].values, np.linspace(0, data.shape[0], data.shape[0]), symb, color=color, alpha=.5, label=label)
        

    data = pd.DataFrame(table.fetch())
    arms = ['Left_Far', 'Left_Medium', 'Centre', 'Right_Medium', 'Right_Far']
    f, axarr = plt.subplots(len(arms), 1)
    axarr = axarr.flatten()

    for arm, ax in zip(arms, axarr):
        escapes = data.loc[(data['arm_taken'] == arm)&(data['is_escape'] == 'true')]
        not_escapes = data.loc[(data['arm_taken'] == arm)&(data['is_escape'] == 'false')]
        
        plotter(ax, escapes.loc[escapes['is_trial'] == 'true'], 'r', 'o', 'trial escape')
        plotter(ax, escapes.loc[escapes['is_trial'] == 'false'], 'r', 'x', 'notrial escape')
        plotter(ax, not_escapes.loc[not_escapes['is_trial'] == 'true'], 'k', 'o', 'trial noescape')
        plotter(ax, not_escapes.loc[not_escapes['is_trial'] == 'false'], 'k', 'x', 'notrial noescape')
        ax.set(title=arm, xlim=[0, 75])
        ax.legend()


def compare_arm_probs(table):
    """
    Plots the probability of taking each arm for all stimulus evoked escapes, spontaneous escapes and spontaneous returns (not escape) for each experiment
    """
    # Get data
    data = pd.DataFrame(table.fetch())

    # Get a set of all arms
    arms_set = set(data['arm_taken'].values)

    # Loop over each experiment
    conditions = namedtuple('conditions', 'is_escape is_trial experiment')
    for experiment in set(list(data['experiment_name'].values)):
        # Get the data for each category
        categories = dict(stim_evoked_escape = conditions('true', 'true', experiment), 
                        spontaneous_escape = conditions('true', 'false', experiment), 
                        stim_evoked_return = conditions('true', 'false', experiment),
                        spontaneous_return = conditions('false', 'false', experiment))

        returns_data = {k:data.loc[(data['is_escape'] == c.is_escape)&(data['is_trial']==c.is_trial)&(data['experiment_name']==c.experiment)]
                        for k,c in categories.items()}

        # Get the arm taken for each return and count the number of trials
        arms_taken = {k:list(v['arm_taken'].values) for k,v in returns_data.items()}
        tot_returns = {k:len(v) for k,v in arms_taken.items()}


        # Get the proportion of returns on each arm
        arms_props = {}
        for arm in arms_set:
            arms_props[arm] = tuple([round(arms_taken[cond].count(arm)/tot_returns[cond], 3)] for cond in categories.keys())

    
        arms_props = pd.DataFrame.from_dict(arms_props)
        print(arms_props)
    

        f, axes = plt.subplots(len(categories.keys()), 1, sharey=True)
        y = np.arange(len(arms_set))
        colors = ['b', 'r', 'g', 'm']
        for i, cat in enumerate(categories.keys()):
            axes[i].bar(y, arms_props.loc[i].values, align='center', color=colors[i], zorder=10)
            if i == 0:
                axes[i].set(title='{} - {} - n={}'.format(experiment, cat, tot_returns[cat]), facecolor=[.2, .2, .2])
            else:
                axes[i].set(title='{} - n={}'.format(cat, tot_returns[cat]), facecolor=[.2, .2, .2])


        for ax in axes.flat:
            ax.margins(0.03)
            ax.grid(True)
            ax.set(ylim=[0, 1])

            ax.set(xticks=y, xticklabels=list(arms_props.keys()))

        f.tight_layout()
        # f.subplots_adjust(wspace=0.09)
    plt.show()


    a=1


if __name__ == "__main__":
    compare_arm_probs(AllTrips())
    plt.show()