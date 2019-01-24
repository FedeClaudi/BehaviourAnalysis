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
        self.selected_experiments_only = True
        self.work_on_experiments = ['Square Maze','PathInt2']

        # Get variables
        self.recording_uid, self.is_trial, self.is_escape, self.experiment_name, self.arm_taken = AllTrips.fetch('recording_uid', 'is_trial', 'is_escape', 'experiment_name', 'arm_taken')
        self.recordings_sessions_lookup, self.sessions_recordings_lookup = self.get_sessions()
        self.experiments = sorted(set(self.experiment_name))
        self.arms = sorted(set(self.arm_taken))

        # Organise data in a dataframe
        self.data = self.create_dataframe()
        self.merge_different_experiments()
        
        # Plot
        self.plot_choiche_by_exp_and_session()

    def get_sessions(self):
        # Get the name of each session that a recording belongs to -    WIP 
        recs_recuid, recs_sesuid = Recordings.fetch('recording_uid', 'uid')

        # Get all recordings for each session
        sessions_recordings_lookup = {}
        for sess in set(recs_sesuid):
            sessions_recordings_lookup[sess] = [r for r,s in zip(recs_recuid, recs_sesuid) if s == sess]
        return {r:s for r,s in zip(recs_recuid, recs_sesuid)}, sessions_recordings_lookup

    def create_dataframe(self):
        sessions = [self.recordings_sessions_lookup[r] for r in self.recording_uid]
        temp_dict = dict(
            session = sessions, 
            recording = self.recording_uid,
            is_trial = self.is_trial,
            is_escape = self.is_escape,
            experiment = self.experiment_name,
            arm = self.arm_taken
        )
        return pd.DataFrame.from_dict(temp_dict)

    def merge_different_experiments(self):
        self.data.loc[self.data['experiment'] == 'TwoAndahalf Maze', 'experiment'] = 'Square Maze'

        sessions_to_change = [135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 161, 162, 163, 164, 165]
        for sess in sessions_to_change:
            self.data.loc[self.data['session'] == sess, 'experiment'] = 'PathInt2'

    def plot_choiche_by_exp_and_session(self):
        def plot_as_barplot():
            
            f, axarr = plt.subplots(2, round(len(experiment_sessions)/2)+1, facecolor=[.2, .2, .2])
            axarr = axarr.flatten()
            for i, session in enumerate(experiment_sessions):
                # Get the data for each session in the experiment
                session_data = selected.loc[(selected['session']==session)&(selected['is_escape']=='true')]

                # Get the probability of escaping on each arm for each session
                session_n_escapes = session_data.shape[0]
                if session_n_escapes != 0: 
                    sesssion_arms = list(session_data['arm'].values)
                    arms_probabilities = {arm: round((sesssion_arms.count(arm)/session_n_escapes), 2) for arm in experiment_arms}

                    # Plot
                    # axarr[i].bar(y, arms_probabilities.values(), color=[.8, .8, .8], align='center', zorder=10)
                    
                
                # Fix axes
                if i == 0:
                    ttl = '{} - {}'.format(exp, session)
                else:
                    ttl = str(session)
                axarr[i].set(title = ttl, facecolor=[.2, .2, .2])
                axarr[i].set(xticks=y, xticklabels=list(arms_probabilities.keys()))

        def plot_as_slope_plot():
            f, axarr = plt.subplots(ncols=3, facecolor=[.2, .2, .2], gridspec_kw = {'width_ratios':[1, 3, 1]})

            _mean = {a:[] for a in experiment_arms}
            all_escapes_counter = 0
            for i, session in enumerate(experiment_sessions):
                # Get the data for each session in the experiment
                session_data = selected.loc[(selected['session']==session)&(selected['is_escape']=='true')]

                # Get the probability of escaping on each arm for each session
                session_n_escapes = session_data.shape[0]
                all_escapes_counter += session_n_escapes
                if session_n_escapes != 0: 
                    sesssion_arms = list(session_data['arm'].values)
                    arms_probabilities = {arm: round((sesssion_arms.count(arm)/session_n_escapes), 2) for arm in experiment_arms}
                    for arm, prob in arms_probabilities.items():
                        _mean[arm].append(prob)
                    axarr[1].plot(arms_probabilities.values(), 'o-', color=[.8, .8, .8], alpha=1)

                    noise = np.random.normal(0,0.5,1)
                    axarr[0].scatter(noise, list(arms_probabilities.values())[0], s=35, c=[.8, .8, .8], alpha=.5)
                    axarr[2].scatter(noise, list(arms_probabilities.values())[1], s=35, c=[.8, .8, .8], alpha=.5)
                
            _mean_to_plot = [np.mean(_mean[arm]) for arm in experiment_arms]
            axarr[1].plot(_mean_to_plot, 'o-', color=[.8, .3, .3], alpha=1, linewidth=2)


            axarr[1].set(title = exp+' # escapes: '+str(all_escapes_counter)+' # sessions: '+str(len(experiment_sessions)), facecolor=[.2, .2, .2])
            axarr[1].set(xticks=y, xticklabels=list(experiment_arms))
            axarr[0].set(facecolor=[.2, .2, .2])
            axarr[2].set(facecolor=[.2, .2, .2])

        for exp in self.experiments:
            if self.selected_experiments_only and not exp in self.work_on_experiments: continue

            # get the data for each experiment 
            selected = self.data.loc[self.data['experiment'] == exp]
            experiment_sessions = set(selected['session'].values)
            experiment_arms = sorted(set(selected['arm'].values))
            if exp == 'PathInt2':
                experiment_arms = ['Left_Far', 'Right_Medium']

            y = np.arange(len(experiment_arms))  # for plotting
            plot_as_slope_plot()
            


if __name__ == "__main__":
    ChoicesVisualiser()

    plt.show()