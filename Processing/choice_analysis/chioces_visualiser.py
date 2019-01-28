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
from database.dj_config import start_connection

from Processing.tracking_stats.math_utils import line_smoother
from Utilities.file_io.files_load_save import load_yaml
from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame






class ChoicesVisualiser:
    def __init__(self, run=True):
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
        
        # Calc p(R) for each mouse
        # self.calc_individuals_pofR()

        # Plot
        if run:
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


    def calc_individuals_pofR(self, experiment):
        selected = self.get_experiment_data(experiment)
        experiment_sessions = set(selected['session'].values)
        experiment_arms = ['Left_Far', 'Right_Medium']

        probabilities = []
        for i, session in enumerate(experiment_sessions):
            # Get the data for each session in the experiment
            session_data = selected.loc[(selected['session']==session)&(selected['is_escape']=='true')]

            if session_data.shape[0] != 0: 
                sesssion_arms = list(session_data['arm'].values)
                # Get the probability of escaping on each arm for each session
                session_n_escapes = len(sesssion_arms)
                if session_n_escapes < 4: continue # ! excluding session with just few escape

                arms_probabilities = {arm: round((sesssion_arms.count(arm)/session_n_escapes), 2) for arm in experiment_arms}

                probabilities.append(arms_probabilities[experiment_arms[-1]])

        return probabilities

    def get_experiment_binary_outcomes(self, experiment):
        selected = self.get_experiment_data(experiment)
        experiment_sessions = set(selected['session'].values)
        experiment_arms = ['Left_Far', 'Right_Medium']
        all_arms_mtx = np.full((len(experiment_sessions), 100), np.nan)
        binary = []
        max_trials_per_session = 0
        for i, session in enumerate(experiment_sessions):
            # Get the data for each session in the experiment
            session_data = selected.loc[(selected['session']==session)&(selected['is_escape']=='true')]
            if session_data.shape[0] != 0: 
                sesssion_arms = list(session_data['arm'].values)
                # Get the probability of escaping on each arm for each session
                session_n_escapes = len(sesssion_arms)
                if session_n_escapes < 4: continue # ! excluding session with just few escape

                arms_binary = [0 if 'Left' in arm else 1 if 'Right' in arm else np.nan for arm in sesssion_arms]
                binary.extend(arms_binary)
                all_arms_mtx[i, :len(arms_binary)] = arms_binary
                if len(arms_binary) > max_trials_per_session: max_trials_per_session = len(arms_binary)

        all_arms_mtx = all_arms_mtx[:, :max_trials_per_session]

        return binary, all_arms_mtx



    def get_experiment_data(self, experiment):
        return self.data[self.data['experiment'] == experiment]
        

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
            f, axarr = plt.subplots(nrows=2, ncols=5, facecolor=[.2, .2, .2], gridspec_kw = {'width_ratios':[1, 2, 6, 2, 1]}, figsize=(12, 10))
            _mean = {a:[] for a in experiment_arms}
            all_escapes_counter = 0
            all_escapes_tracker = []
            all_arms_counter = {arm:0 for arm in experiment_arms}
            all_arms_mtx = np.full((len(experiment_sessions), 100), np.nan)
            for i, session in enumerate(experiment_sessions):
                # Get the data for each session in the experiment
                session_data = selected.loc[(selected['session']==session)&(selected['is_escape']=='true')]

                if session_data.shape[0] != 0: 
                    sesssion_arms = list(session_data['arm'].values)
                    # Get the probability of escaping on each arm for each session
                    session_n_escapes = len(sesssion_arms)
                    if session_n_escapes < 4: continue # ! excluding session with just few escape
                    all_escapes_tracker.append(session_n_escapes)
                    all_escapes_counter += session_n_escapes
                    arms_probabilities = {arm: round((sesssion_arms.count(arm)/session_n_escapes), 2) for arm in experiment_arms}

                    arms_binary = [0 if 'Left' in arm else 1 if 'Right' in arm else np.nan for arm in sesssion_arms]
                    all_arms_mtx[i, :len(arms_binary)] = arms_binary
                    for arm in experiment_arms:
                        all_arms_counter[arm] += sesssion_arms.count(arm)

                    for arm, prob in arms_probabilities.items():
                        _mean[arm].append(prob)
                    axarr[0, 2].plot(arms_probabilities.values(), 'o-', color=[.6, .6, .8], alpha=.75)

                    noise = np.random.normal(5,1,1)
                    axarr[0, 1].scatter(noise, list(arms_probabilities.values())[0], s=session_n_escapes*5, c=[.6, .6, .8], alpha=.75)
                    axarr[0, 3].scatter(noise, list(arms_probabilities.values())[1], s=session_n_escapes*5, c=[.6, .6, .8], alpha=.75)

                    # axarr[1, 1].scatter(session_n_escapes,  list(arms_probabilities.values())[1])
                
            axarr[0, 0].hist(list(_mean.values())[0], bins=20, histtype='stepfilled', orientation='horizontal', color=[.5, .5, .8], density=False, alpha=.5)
            axarr[0, 4].hist(list(_mean.values())[1], bins=20, histtype='stepfilled', orientation='horizontal', color=[.5, .5, .8], density=False, alpha=.5)

            _mean_to_plot = [np.average(_mean[arm], weights=all_escapes_tracker) for arm in experiment_arms]
            _std_to_plot = [np.std(_mean[arm]) for arm in experiment_arms]
            _sem_to_plot = [stats.sem(_mean[arm]) for arm in experiment_arms]
            x = [0, 1]
            axarr[0, 2].errorbar(x, _mean_to_plot, yerr=_sem_to_plot, color=[.8, .3, .3], alpha=1, linewidth=3, elinewidth =1, label='Weighted avg. of individuals probs')

            comulative_probs = {arm:count/all_escapes_counter for arm,count in all_arms_counter.items()}
            # axarr[1].plot(comulative_probs.values(), 'o-', color=[.3, .8, .3], alpha=.75, linewidth=3, label='Comulative probability')

            # Plot each trial for all mice
            sort_by_r_prob = np.argsort(np.nanmean(all_arms_mtx, 1))
            axarr[1, 2].imshow(all_arms_mtx[np.array(sort_by_r_prob), :], cmap='Blues')

            # Plot distribution of p(R)
            mu, variance = _mean_to_plot[1], _std_to_plot[1]
            sigma = math.sqrt(variance)
            x = np.linspace(mu - variance, mu + variance, 100)
            axarr[1, 1].fill_between(stats.norm.pdf(x, loc=mu), 0, x, color=[.6, .6, .8], alpha=.6)

            for ax in axarr[0, :]:
                ax.set(facecolor=[.2, .2, .2])
                ax.axhline(0, linestyle='--', color=[.9, .9, .9], linewidth=.5)
                ax.axhline(0.5, linestyle='--', color=[.9, .9, .9], linewidth=.5)
                ax.axhline(1, linestyle='--', color=[.9, .9, .9], linewidth=.5)

            for axn, ax in enumerate(axarr[1, :]):
                ax.set(facecolor=[.2, .2, .2])
                if axn not in [1, 2]:
                    ax.set(xticks=[], yticks=[])



            axarr[0, 0].set(ylim=[-0.1, 1.1], title='# escapes: ')
            axarr[0, 1].set(ylim=[-0.1, 1.1], title = 'min-{}, max-{}'.format(min(all_escapes_tracker), max(all_escapes_tracker)), yticks=[],  xticks=[])
            axarr[0, 2].set( ylim=[-0.1, 1.1])
            axarr[0, 2].set_title(exp+' # escapes: '+str(all_escapes_counter)+' # sessions: {}/{}'.format(len(all_escapes_tracker), i), color=[.3, .8, .3])
            axarr[0, 2].set(xticks=y, xticklabels=list(experiment_arms), yticks=[])
            axarr[0, 2].legend(fancybox=True, framealpha=0.5)
            axarr[0, 3].set(ylim=[-0.1, 1.1],  title = '# escapes: ', yticks=[],  xticks=[] )
            axarr[0, 4].set( ylim=[-0.1, 1.1], yticks=[] , title=' mean-{}, std-{}'.format(round(np.mean(all_escapes_tracker), 2), round(np.std(all_escapes_tracker), 2)))
            axarr[0, 4].set_xlim(axarr[0, 4].get_xlim()[::-1])

            axarr[1, 1].set(ylim=[0, 1])
            axarr[1, 2].set(xlim=[0, max(all_escapes_tracker)], title='All trials per mouse')


            plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)


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