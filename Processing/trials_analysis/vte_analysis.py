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
from scipy.integrate import cumtrapz as integral
import seaborn as sns
from tqdm import tqdm
import random

import multiprocessing as mp


mpl.rcParams['text.color'] = 'k'
mpl.rcParams['xtick.color'] = 'k'
mpl.rcParams['ytick.color'] = 'k'
mpl.rcParams['axes.labelcolor'] = 'k'

from database.NewTablesDefinitions import *

from Processing.tracking_stats.math_utils import *
from Utilities.file_io.files_load_save import load_yaml
from Processing.plot.tracking_onmaze_videomaker import VideoMaker

from database.database_fetch import *



class VTE:
    def __init__(self):
        self.zscore_th = .5
        self.video_maker = VideoMaker()

    """
        =======================================================================================================================================================
            TABLE FUNCTIONS    
        =======================================================================================================================================================
    """

    def drop(self):
        ZidPhi.drop()
        sys.exit()

    def populate(self):
        zdiphi = ZidPhi()
        zdiphi.populate()

    @staticmethod
    def populate_vte_table(table, key, max_y = 450, displacement_th = 20, min_length=20):

        from database.NewTablesDefinitions import AllTrials

        # get tracking from corresponding trial
        trials = (AllTrials & "trial_id={}".format(key['trial_id']) & "is_escape='true'").fetch("tracking_data","session_uid", "experiment_name", "escape_arm")
        try:
            trials[0][0]
        except:
            # no trials
            return

        print("processing trial id: ", key['trial_id'])

        tracking, uid, experiment, escape_arm = trials[0][0], trials[1][0], trials[2][0], trials[3][0]
        # get xy for snout and platform at each frame
        x, y, platf = median_filter(tracking[:, 0, 1]), median_filter(tracking[:, 1, 1]),  median_filter(tracking[:, -1, 0])

        try:
            # Select the times between when the mice leave the catwalk and when they leave the threat platform
            # first_above_catwalk = np.where(y > 250)[0][0]
            first_above_catwalk = 0
            first_out_threat = np.where(platf != 1)[0][0]
        except:
            return

        n_steps = first_out_threat - first_above_catwalk
        if n_steps < min_length: return  # the trial started too close to max Y

        # using interpolation to remove nan from array
        x, y = fill_nans_interpolate(x[first_above_catwalk : first_out_threat]), fill_nans_interpolate(y[first_above_catwalk : first_out_threat])

        dx, dy = np.diff(x), np.diff(y)

        try:
            dphi = calc_angle_between_points_of_vector(np.vstack([dx, dy]).T)
        except:
            raise ValueError
        idphi = np.trapz(dphi)

        key['xy'] = np.vstack([x, y])
        key['dphi'] = dphi
        key['idphi'] = idphi
        key['session_uid'] = uid
        key['experiment_name'] = experiment
        key['escape_arm'] = escape_arm

        table.insert1(key)


    """
        =======================================================================================================================================================
            PLOTTING FUNCTIONS    
        =======================================================================================================================================================
    """



    def zidphi_histogram(self, experiment=None, title=''):
        if experiment is None:
            idphi = ZidPhi.fetch('idphi')
        else:
            if isinstance(experiment, str):
                idphi = (ZidPhi & "experiment_name='{}'".format(experiment)).fetch('idphi')
            else:
                idphi = []
                for exp in experiment:
                    i = (ZidPhi & "experiment_name='{}'".format(exp)).fetch('idphi')
                    idphi.extend(i)

        zidphi = stats.zscore(idphi)

        above_th = [1 if z>self.zscore_th else 0 for z in zidphi]
        perc_above_th = np.mean(above_th)*100

        f, ax = plt.subplots()

        ax.hist(zidphi, bins=16, color=[.4, .7, .4])
        ax.axvline(self.zscore_th, linestyle=":", color='k')
        ax.set(title=title+" {}% VTE".format(round(perc_above_th, 2)), xlabel='zIdPhi')


    def zidphi_tracking(self, experiment=None, title=''):
        if experiment is None:
            trials = ZidPhi.fetch('idphi', "xy")
        else:
            if isinstance(experiment, str):
                trials = (ZidPhi & "experiment_name='{}'".format(experiment)).fetch('idphi', "xy")
            else:
                trials = []
                for exp in experiment:
                    t = (ZidPhi & "experiment_name='{}'".format(exp)).fetch('idphi', "xy")
                    trials.extend(t)

        data = pd.DataFrame.from_dict(dict(idphi=trials[0], xy=trials[1]))
        data['zidphi'] = stats.zscore(data['idphi'].values)

        f, axarr = plt.subplots(ncols=2)

        for i, row in data.iterrows():
            if row['zidphi'] > self.zscore_th:
                axn = 1
            else:
                axn = 0

            axarr[axn].plot(row['xy'].T[:, 0], row['xy'].T[:, 1], alpha=.3, linewidth=2)

        axarr[0].set(title=title + ' non VTE trials', ylabel='Y', xlabel='X')
        axarr[1].set(title='VTE trials', ylabel='Y', xlabel='X')



    """
        =======================================================================================================================================================
            VIDEO FUNCTIONS    
        =======================================================================================================================================================
    """


    def zidphi_videos(self, experiment=None, title='', fps=30, background='', vte=True):
        trials = []
        for exp in experiment:
            t = (ZidPhi & "experiment_name='{}'".format(exp)).fetch('idphi', "trial_id")
            trials.extend(t)

        data = pd.DataFrame.from_dict(dict(idphi=trials[0], trial_id=trials[1]))
        data['zidphi'] = stats.zscore(data['idphi'].values)

        data['rec_uid'] = [(AllTrials & "trial_id={}".format(i)).fetch("recording_uid")[0] for i in data['trial_id']]
        data['tracking'] = [(AllTrials & "trial_id={}".format(i)).fetch("tracking_data")[0] for i in data['trial_id']]

        data['origin'] = ['' for i in np.arange(len(data['rec_uid']))]
        data['escape'] = ['' for i in np.arange(len(data['rec_uid']))]
        data['stim_frame'] = ['' for i in np.arange(len(data['rec_uid']))]

        if vte:
            data = data.loc[data['zidphi'] > self.zscore_th]
        else:
            data = data.loc[data['zidphi'] <= self.zscore_th]

        self.video_maker.data = data
        self.video_maker.make_video(videoname = title, experimentname=background, fps=fps, 
                                    savefolder=self.video_maker.save_fld_trials, trial_mode=False)

    def parallel_videos(self):
        a1 = (['PathInt2', 'PathInt2 - L'], "Asymmetric Maze - NOT VTE", 40, 'PathInt2', False)
        a2 = (['PathInt2', 'PathInt2 - L'], "Asymmetric Maze - VTE", 40, 'PathInt2', True)
        a3 = (['Square Maze', 'TwoAndahalf Maze'], "Symmetric Maze - NOT VTE", 40, 'Square Maze', False)
        a4 = (['Square Maze', 'TwoAndahalf Maze'], "Symmetric Maze - VTE", 40, 'Square Maze', True)

        a = [a1, a2, a3, a4]

        processes = [mp.Process(target=self.zidphi_videos, args=arg) for arg in a]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

    def parallel_videos2(self):
        a1 = (['Model based'], "MB - NOT VTE", 40, 'Model Based', False)
        a2 = (['Model Based', 'PathInt2 - L'], "MB - VTE", 40, 'Model Based', True)


        a = [a1, a2]

        processes = [mp.Process(target=self.zidphi_videos, args=arg) for arg in a]

        for p in processes:
            p.start()

        for p in processes:
            p.join()
        


    """
        =======================================================================================================================================================
            STATS FUNCTIONS    
        =======================================================================================================================================================
    """

    def pR_byVTE(self, experiment=None, title=None):
        trials = []
        for exp in experiment:
            t = (ZidPhi & "experiment_name='{}'".format(exp)).fetch('idphi', "escape_arm")
            trials.extend(t)

        data = pd.DataFrame.from_dict(dict(idphi=trials[0], escape_arm=trials[1]))
        data['zidphi'] = stats.zscore(data['idphi'].values)

        overall_pR = calc_prob_item_in_list(list(data['escape_arm'].values), 'Right_Medium')

        non_vte_pR = calc_prob_item_in_list(list(data.loc[data['zidphi'] < self.zscore_th]['escape_arm'].values), 'Right_Medium')
        vte_pR = calc_prob_item_in_list(list(data.loc[data['zidphi'] >= self.zscore_th]['escape_arm'].values), 'Right_Medium')

        print("""
        Experiment {}
                overall pR: {}
                VTE pR:     {}
                non VTE pR: {}
        
        """.format(title, round(overall_pR, 2), round(vte_pR, 2), round(non_vte_pR, 2)))

        n_vte_trials = len(list(data.loc[data['zidphi'] >= self.zscore_th]['escape_arm'].values))
        random_pR = []
        for i in np.arange(100000):
            random_pR.append(calc_prob_item_in_list(random.choices(list(data['escape_arm'].values), k=n_vte_trials), 'Right_Medium'))


        f, ax = plt.subplots()
        ax.hist(random_pR, bins=30, color=[.4, .7, .4], density=True)
        ax.axvline(overall_pR, color='k', linestyle=':', label='Overall p(R)', linewidth=3)
        ax.axvline(vte_pR, color='r', linestyle=':', label='VTE p(R)', linewidth=3)
        ax.axvline(non_vte_pR, color='g', linestyle=':', label='nVTE p(R)', linewidth=3)
        ax.set(title=title)
        ax.legend()


"""
    =======================================================================================================================================================
    =======================================================================================================================================================
    =======================================================================================================================================================
    =======================================================================================================================================================
"""
if __name__ == "__main__":
    vte = VTE()

    # vte.drop()
    # vte.populate()

    vte.pR_byVTE(experiment=['PathInt2', 'PathInt2-L'], title="Asymmetric Maze")
    vte.pR_byVTE(experiment=['Square Maze', 'TwoAndahalf Maze'], title='Symmetric Maze')
    vte.pR_byVTE(experiment=[ 'PathInt2-D'], title="Asymmetric Maze Dark")

    # vte.zidphi_histogram(experiment=['PathInt2', 'PathInt2 - L'], title="Asymmetric Maze")
    # vte.zidphi_histogram(experiment=['Square Maze', 'TwoAndahalf Maze'], title='Symmetric Maze')
    # vte.zidphi_histogram(experiment=['Model Based'], title="Model Based")
    # vte.zidphi_histogram(experiment=['FourArms Maze'], title="4 arm")


    # vte.zidphi_tracking(experiment=['PathInt2', 'PathInt2 - L'], title="Asymmetric Maze")
    # vte.zidphi_tracking(experiment=['Square Maze', 'TwoAndahalf Maze'], title='Symmetric Maze')
    # vte.zidphi_tracking(experiment=['Model Based'], title="Model Based")
    # vte.zidphi_tracking(experiment=['FourArms Maze'], title="4 arm")


    # vte.parallel_videos()
    # vte.parallel_videos2()



    plt.show()
