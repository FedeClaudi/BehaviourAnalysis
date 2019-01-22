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


def compare_arm_probs(table):
    """
    Plots the probability of taking each arm for all stimulus evoked escapes, spontaneous escapes and spontaneous returns (not escape) for each experiment
    """
    # Get data
    data = pd.DataFrame(table.fetch())


    # Merge squared maze and two and a half maze datasets
    data.loc[data['experiment_name'] == 'TwoAndahalf Maze', 'experiment_name'] = 'Square Maze'
    # data.loc[data['experiment_name'] == 'FlipFlop Maze', 'experiment_name'] = 'PathInt2'


    # Get a set of all arms
    arms_set = sorted(set(data['arm_taken'].values))
    # arms_set = ('Left_Far', 'Right_Medium', 'Centre')

    # Loop over each experiment
    conditions = namedtuple('conditions', 'is_escape is_trial experiment')
    for experiment in set(list(data['experiment_name'].values)):
        # Get the data for each category
        categories = dict(stim_evoked_escape = conditions('true', 'true', experiment), 
                        spontaneous_escape = conditions('true', 'false', experiment), 
                        stim_evoked_return = conditions('false', 'true', experiment),
                        spontaneous_return = conditions('false', 'false', experiment),
                        all_return = conditions('false', ['true', 'false'], experiment),
                        all_escape = conditions('true', ['true', 'false'], experiment))


        returns_data = {} # = {k:data.loc[(data['is_escape'] == c.is_escape)&(data['is_trial'] in c.is_trial)&(data['experiment_name']==c.experiment)]
                            #for k,c in categories.items()}

        for name, c in categories.items():
            if not isinstance(c.is_trial, list):
                returns_data[name] = data.loc[(data['is_escape'] == c.is_escape)&(data['is_trial'] == c.is_trial)&(data['experiment_name']==c.experiment)]
            else:
                returns_data[name] = data.loc[(data['is_escape'] == c.is_escape)&(data['experiment_name']==c.experiment)]


        # Get the arm taken for each return and count the number of trials
        arms_taken = {k:list(v['arm_taken'].values) for k,v in returns_data.items()}
        tot_returns = {k:len(v) for k,v in arms_taken.items()}

        # Get median duration per arm per condition
        durations = {}
        for k,v in returns_data.items():
            durations[k] = tuple([np.mean(v.loc[v['arm_taken']==arm]['duration'].values) for arm in arms_set])

        # Get the proportion of returns on each arm
        arms_props = {}
        for arm in arms_set:
            arms_props[arm] = tuple([round(arms_taken[cond].count(arm)/tot_returns[cond], 3)] for cond in categories.keys())
    
        arms_props = pd.DataFrame.from_dict(arms_props)
        print(arms_props)

        f, axes = plt.subplots(len(categories.keys()), 2, facecolor=[.2, .2, .2])
        y = np.arange(len(arms_set))
        colors = ['b', 'r', 'g', 'm', 'y', 'c']

        for i, cat in enumerate(categories.keys()):
            axes[i, 0].bar(y, arms_props.loc[i].values, align='center', color=colors[i], zorder=10)
            axes[i, 1].bar(y, durations[cat], align='center', color=colors[i], zorder=10)
            if i == 0:
                axes[i, 0].set(title='{} - {} - n={}'.format(experiment, cat, tot_returns[cat]), facecolor=[.2, .2, .2], ylim=[0, 1])
            else:
                axes[i, 0].set(title='{} - n={}'.format(cat, tot_returns[cat]), facecolor=[.2, .2, .2], ylim=[0, 1])
            
            axes[i, 1].set(title='median duration (s)', facecolor=[.2, .2, .2], ylim=[0, 25])


        

        for ax in axes.flat:
            ax.margins(0.03)
            ax.grid(True)
            

            ax.set(xticks=y, xticklabels=list(arms_props.keys()))

        f.tight_layout()
        # f.subplots_adjust(wspace=0.09)
    plt.show()


    a=1


if __name__ == "__main__":
    compare_arm_probs(AllTrips())
    plt.show()