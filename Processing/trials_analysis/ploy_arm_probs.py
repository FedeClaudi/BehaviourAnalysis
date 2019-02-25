import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from database.NewTablesDefinitions import *
from database.database_fetch import *

from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame, get_arm_given_rois, convert_roi_id_to_tag
from Processing.tracking_stats.math_utils import get_roi_enters_exits, line_smoother, calc_distance_between_points_2d, remove_tracking_errors




class Plotter:
    def __init__(self):
        return

    @staticmethod
    def calc_arm_p(escapes, target):
        escapes = list(escapes)
        if not escapes: return 0
        return escapes.count(target)/len(escapes)

    def plot_pR_by_exp(self):
        target = 'Right_Medium'
        escapes = {}
        experiments = set(AllTrials.fetch('experiment_name'))
        for exp in sorted(experiments):           
            escape = get_trials_by_exp(exp, 'true', ['escape_arm'])
            escapes[exp] = self.calc_arm_p(escape, target)


        x = np.arange(len(experiments))

        f, ax = plt.subplots()
        ax.bar(x, escapes.values(), color=[.4, .4, .4], align='center', zorder=10)
        ax.set(title="$p(R)$", xticks=x, xticklabels=escapes.keys())



    def plot_pR_individuals_by_exp(self):
        target = 'Right_Medium'
        escapes = {}
        experiments = set(AllTrials.fetch('experiment_name'))

        experiments_pR = {}
        for exp in sorted(experiments):         
            sessions = get_trials_by_exp(exp, 'true', ['session_uid'])
            experiment_pr = []
            for uid in set(sessions):
                escape = get_trials_by_exp_and_session(exp, uid, 'true', ['escape_arm'])
                pR = self.calc_arm_p(escape, target)
                experiment_pr.append(pR)

            experiments_pR[exp] = np.array(experiment_pr)


        x = np.arange(len(experiments))

        f, ax = plt.subplots()
        for n, pR in enumerate(experiments_pR.values()):
            xx = np.random.normal(n, 0.1, len(pR))
            ax.scatter(xx, pR, s=90, alpha=.25)
            ax.scatter(n, np.mean(pR), color='r', s=120, alpha=1)
            # ax.axvline(n,  color='r', linewidth=.5)
            ax.axhline(np.mean(pR),  color='r', linewidth=.5)

        ax.axhline(.5, color='k', linewidth=.5)
        ax.set(title="$p(R)$", xticks=x, xticklabels=experiments_pR.keys())


if __name__ == "__main__":
    p = Plotter()
    p.plot_pR_individuals_by_exp()


    plt.show()