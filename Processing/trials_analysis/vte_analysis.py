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


from database.NewTablesDefinitions import *

from Processing.tracking_stats.math_utils import *
from Utilities.file_io.files_load_save import load_yaml
from Processing.plot.tracking_onmaze_videomaker import VideoMaker
from Processing.trials_analysis.tc_plotting import plot_two_dists_kde
from Processing.modelling.bayesian.hierarchical_bayes_v2 import Modeller as Bayes
from database.database_fetch import *



class VTE:
    def __init__(self):
        self.zscore_th = 0.5
        self.video_maker = VideoMaker()
        self.bayes = Bayes()

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
        sys.exit()

    @staticmethod
    def populate_vte_table(table, key, max_y = 450, displacement_th = 20, min_length=10):

        from database.NewTablesDefinitions import AllTrials

        # get tracking from corresponding trial
        trials = (AllTrials & "trial_id={}".format(key['trial_id']) & "is_escape='true'" ).fetch("tracking_data","session_uid", "experiment_name", "escape_arm", "origin_arm")
        try:
            trials[0][0]
        except:
            # no trials
            return

        print("processing trial id: ", key['trial_id'])

        tracking, uid, experiment, escape_arm, origin_arm = trials[0][0], trials[1][0], trials[2][0], trials[3][0],  trials[4][0]
        # get xy for snout and platform at each fram - interpolate to remove nan
        x, y, platf = tracking[:, 0, 1], tracking[:, 1, 1],  tracking[:, -1, 0]
        bx, by = tracking[:, 0, 0], tracking[:, 1, 0]

        try:
            # Select the times before when they leave the threat platform
            first_out_threat = np.where(platf != 1)[0][0]

            # Selct the time when the mouse moved up for N pixels
            # moved_up = np.where(y>= np.nanmean(y[0:10]) + displacement_th)[0][0]
        except:
            return
        x = x[0:first_out_threat]
        y = y[0:first_out_threat]

        # Calc iDiPhi
        fps = get_videometadata_given_recuid(get_recs_given_sessuid(uid)['recording_uid'][0])
        if fps == 0: fps = 40

        dx, dy = np.diff(x), np.diff(y)
        dphi = np.rad2deg(np.diff([math.atan2(xx,yy) for xx,yy in zip(dx, dy)]))
        idphi = np.trapz(np.abs(dphi), dx=1000/fps)

        # Calculate speed of movement
        try:
            s = calc_distance_between_points_in_a_vector_2d(np.array([x,y]).T) * fps
        except:
            pass
        else:

            key['xy'] = np.vstack([x, y, s])
            key['dphi'] = dphi
            key['idphi'] = idphi
            key['session_uid'] = uid
            key['experiment_name'] = experiment
            key['escape_arm'] = escape_arm
            key['origin_arm'] = origin_arm

            table.insert1(key)


    """
        =======================================================================================================================================================
            PLOTTING FUNCTIONS    
        =======================================================================================================================================================
    """

    def get_dataframe(self, experiment):
        data = pd.DataFrame(ZidPhi.fetch())
        data['zidphi'] = stats.zscore(data['idphi'].values)
        if experiment is not None:
            data = data[data['experiment_name'].isin(experiment)]
        return data



    def zidphi_histogram(self, experiment=None, title='', ax=None):
        data = self.get_dataframe(experiment)
        zidphi = stats.zscore(data['idphi'])

        above_th = [1 if z>self.zscore_th else 0 for z in zidphi]
        perc_above_th = np.mean(above_th)*100

        ax.hist(zidphi, bins=16, color=[.4, .7, .4])
        ax.axvline(self.zscore_th, linestyle="--", color='k', linewidth=2)
        ax.axvline(np.percentile(zidphi, 5), linestyle=":", color='k')
        ax.axvline(np.percentile(zidphi, 95), linestyle=":", color='k')

        ax.set(title=title+" {}% VTE".format(round(perc_above_th, 2)), xlabel='zIdPhi')


    def zidphi_tracking(self, experiment=None, title='', axarr=None):
        data = self.get_dataframe(experiment)

        for i, row in data.iterrows():
            if row['zidphi'] > self.zscore_th:
                axn = 1
            else:
                axn = 0

            axarr[axn].plot(row['xy'].T[:, 0], row['xy'].T[:, 1], alpha=.3, linewidth=2)

        axarr[0].set(title=title + ' non VTE trials', ylabel='Y', xlabel='X')
        axarr[1].set(title='VTE trials', ylabel='Y', xlabel='X')


    def zidphi_tracking_examples(self, experiment=None, title='', axarr=None):
        data = self.get_dataframe(experiment)

        low = np.percentile(data['zidphi'].values,5)
        high = np.percentile(data['zidphi'].values, 95)

        for i, row in data.iterrows():
            if row['zidphi'] < low:
                axn = 0
            elif row['zidphi'] > high:
                axn = 1
            else:
                continue

            axarr[axn].plot(row['xy'].T[:, 0], row['xy'].T[:, 1], alpha=.3, linewidth=2)

        axarr[0].set(title=' low zidphi', ylabel='Y', xlabel='X')
        axarr[1].set(title='high zidphi', ylabel='Y', xlabel='X')


    def vte_position(self, experiment=None, title='', background=''):
        data = self.get_dataframe(experiment)

        template = get_maze_template(background)

        f, ax = plt.subplots()
        ax.imshow(template)
        for i, row in data.iterrows():
            if row['zidphi'] > self.zscore_th:
                axn = 1
                c = 'r'
            else:
                axn = 0
                c = 'g'
            
            ax.scatter(row['xy'].T[0, 0], row['xy'].T[0, 1], c=c, alpha=.6, s=80)
        ax.set(title=title + ' non VTE trials', ylabel='Y', xlabel='X', xlim=[400, 600], ylim=[0, 400])


    def vte_speed(self, experiment=None, title='', background='', ax=None):
        data = self.get_dataframe(experiment)
        data['tracking'] = [(AllTrials & "trial_id={}".format(i)).fetch("tracking_data")[0] for i in data['trial_id']]
        data['recording_uid'] = [(AllTrials & "trial_id={}".format(i)).fetch("recording_uid")[0] for i in data['trial_id']]

        vte_speed, n_vte_speed = [], []
     
        for i, row in data.iterrows():
            # Take the mean speed from when they leave the arm to when they get to the shetler
            fps = get_videometadata_given_recuid(row['recording_uid'])

            whet_out_of_t = np.where(row['tracking'][:, -1, 0] != 1)[0][0]
            mean_speed = np.nanmean(row['tracking'][whet_out_of_t:, 2, 0] * fps)

            if row['zidphi'] > self.zscore_th:
                x = np.random.normal(0, .05, 1)
                c = 'r'
                vte_speed.append(mean_speed)
            else:
                x = np.random.normal(1, .05, 1)
                c = 'g'
                n_vte_speed.append(mean_speed)


            ax.scatter(x, mean_speed, c=c, alpha=.8, s=30)
            
        try:
            vte_CI, vte_range = mean_confidence_interval(vte_speed), percentile_range(vte_speed)
            nvte_CI, nvte_range = mean_confidence_interval(n_vte_speed), percentile_range(n_vte_speed)
        except:
            return

        ax.plot([0.25, 0.25], [vte_CI.interval_min, vte_CI.interval_max], color='r', linewidth=4)
        ax.plot([0.25, 0.25], [vte_range.low, vte_range.high], color='r', linewidth=2)

        ax.plot([.75, .75], [nvte_CI.interval_min, nvte_CI.interval_max], color='g', linewidth=4)
        ax.plot([.75, .75], [nvte_range.low, nvte_range.high], color='g', linewidth=2)

        ax.set(title=title , ylabel='Speed', xticks=[0, 1], xticklabels=['VTE', 'Not VTE'])


    """
        =======================================================================================================================================================
            VIDEO FUNCTIONS    
        =======================================================================================================================================================
    """


    def zidphi_videos(self, experiment=None, title='', fps=30, background='', vte=True):
        data = self.get_dataframe(experiment)

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


    def make_parall_videos(self, arguments):
        processes = [mp.Process(target=self.zidphi_videos, args=arg) for arg in arguments]

        for p in processes:
            p.start()

        for p in processes:
            p.join()


    def parallel_videos(self):
        a1 = (['PathInt2', 'PathInt2-L'], "Asymmetric Maze - NOT VTE", 40, 'PathInt2', False)
        a2 = (['PathInt2', 'PathInt2-L'], "Asymmetric Maze - VTE", 40, 'PathInt2', True)
        a3 = (['Square Maze', 'TwoAndahalf Maze'], "Symmetric Maze - NOT VTE", 40, 'Square Maze', False)
        a4 = (['Square Maze', 'TwoAndahalf Maze'], "Symmetric Maze - VTE", 40, 'Square Maze', True)
        self.make_parall_videos([a1, a2, a3, a4])

        

    def parallel_videos2(self):
        a1 = (['Model based'], "MB - NOT VTE", 40, 'Model Based', False)
        a2 = (['Model Based', 'PathInt2 - L'], "MB - VTE", 40, 'Model Based', True)
        self.make_parall_videos([a1, a2])


    

    """
        =======================================================================================================================================================
            STATS FUNCTIONS    
        =======================================================================================================================================================
    """

    def pR_byVTE(self, experiment=None, title=None, target="Right_Medium", ax=None, bayes=True):
        data = self.get_dataframe(experiment)
        overall_escapes = [1 if e == target else 0 for e in list(data['escape_arm'].values)]
        vte_escapes = [1 if e == target else 0 for e in list(data.loc[data['zidphi'] >= self.zscore_th]['escape_arm'].values)]
        non_vte_escapes = [1 if e == target else 0 for e in list(data.loc[data['zidphi'] < self.zscore_th]['escape_arm'].values)]

        overall_pR = np.mean(overall_escapes)
        non_vte_pR = np.mean(non_vte_escapes)
        vte_pR = np.mean(vte_escapes)

        print("""
        Experiment {}
                overall pR: {}
                VTE pR:     {}
                non VTE pR: {}
        
        """.format(title, round(overall_pR, 2), round(vte_pR, 2), round(non_vte_pR, 2)))

        # boot strap
        n_vte_trials = len(list(data.loc[data['zidphi'] >= self.zscore_th]['escape_arm'].values))
        random_pR = []
        for i in np.arange(100000):
            random_pR.append(np.mean(random.choices(overall_escapes, k=n_vte_trials)))

        # plot with bayes
        if bayes:
            try:
                vte_vs_non_vte, _, _, _, _ = self.bayes.model_two_distributions(vte_escapes, non_vte_escapes)
                vte_vs_all, _, _, _, _ = self.bayes.model_two_distributions(vte_escapes, overall_escapes)
                plot_two_dists_kde(vte_vs_all['p_d2'], vte_vs_all['p_d1'], vte_vs_non_vte['p_d2'], title,   l1="VTE", l2="not VTE", ax=ax)

                
            except:
                return

    def pR_bysess_VTE(self, experiment=None, title=None, target="Right_Medium", ax=None):
        data = self.get_dataframe(experiment)

        has_vte = 0
        for sess in set(sorted(data['session_uid'].values)):
            sess_data = data.loc[data['session_uid']==sess]
            y =  np.random.normal(0, 0.05, 1)
            pR_VTE = np.nanmean([1 if e == target else 0 for e in list(sess_data.loc[sess_data['zidphi'] >= self.zscore_th]['escape_arm'].values)]) 
            pR_nVTE = np.nanmean([1 if e == target else 0 for e in list(sess_data.loc[sess_data['zidphi'] < self.zscore_th]['escape_arm'].values)]) 

            if pR_VTE == pR_VTE and pR_nVTE == pR_nVTE:  # ? check that they are both not nan
                has_vte += 1

                ax.plot([0, 1], [pR_nVTE, pR_VTE], 'o', color='k')
                ax.plot([0, 1], [pR_nVTE, pR_VTE],  color='k')
                ax.scatter( 3, pR_VTE - pR_nVTE, alpha=.5, s=50, color='k')

        n_mice = len(set(sorted(data['session_uid'].values)))
        perc_with_vte = round((has_vte / n_mice)*100, 2)
        ax.set(title=title + " - {}% of {} mice with VTE".format(perc_with_vte, n_mice), ylabel='$p(R)$', xticks=[0, 1, 3], xticklabels=['Not VTE', 'VTE', 'd(p(R))'])

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

    # vte.parallel_videos()



    experiments_to_plot =  ['PathInt2', 'PathInt2-L'],  ['Square Maze', 'TwoAndahalf Maze'],
    titles = [ 'Asymmetric',  'Symmetric']
    backgrounds = ['PathInt2', 'Square Maze']

    for e,t,bg in zip(experiments_to_plot, titles, backgrounds):
        f, axarr = plt.subplots(ncols=4, nrows=2)
        vte.pR_bysess_VTE(experiment=e, title='p(R|VTE)', ax=axarr[1, 1])
        vte.vte_speed(experiment=e, title='Mean Speed ', background=bg, ax=axarr[0, 1])
        vte.pR_byVTE(experiment=e, title='p(R|VTE)', ax=axarr[1, 0], bayes=True)
        vte.zidphi_histogram(experiment=e, title=t + ' zdiphi ', ax=axarr[0, 0])
        vte.zidphi_tracking(experiment=e, title='tracking', axarr=axarr[:, 2])
        vte.zidphi_tracking_examples(experiment=e, title='tracking - examples', axarr=axarr[:, 3])
        break
      
    plt.show()
