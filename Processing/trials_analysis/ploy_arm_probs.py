import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from database.NewTablesDefinitions import *
from database.database_fetch import *

from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame, get_arm_given_rois, convert_roi_id_to_tag
from Processing.tracking_stats.math_utils import get_roi_enters_exits, line_smoother, calc_distance_between_points_2d, remove_tracking_errors, get_n_colors

from Processing.modelling.bayesian.hierarchical_bayes_v2 import Modeller as Bayesian


class Plotter:
    def __init__(self):
        self.experiments = set(AllTrials.fetch('experiment_name'))
        self.n_experiments = len(self.experiments)

    @staticmethod
    def calc_arm_p(escapes, target):
        escapes = list(escapes)
        if not escapes: return 0
        return escapes.count(target)/len(escapes)

    def plot_pR_by_exp(self):
        target = 'Right_Medium'
        escapes = {}
        for exp in sorted(self.experiments):           
            escape = get_trials_by_exp(exp, 'true', ['escape_arm'])
            escapes[exp] = self.calc_arm_p(escape, target)


        x = np.arange(self.n_experiments)

        f, ax = plt.subplots()
        ax.bar(x, escapes.values(), color=[.4, .4, .4], align='center', zorder=10)
        ax.set(title="$p(R)$", xticks=x, xticklabels=escapes.keys())


    def plot_p_same_origin_and_escape_by_exp(self):
        probs = {}
        for exp in sorted(self.experiments):         
            sessions = get_trials_by_exp(exp, 'true', ['session_uid'])

            exp_probs = []
            for uid in set(sessions):
                origins, escapes = get_trials_by_exp_and_session(exp, uid, 'true', ['origin_arm', 'escape_arm'])
                samezies = [1 if o == e else 0 for o,e in zip(origins, escapes)]
                exp_probs.append(np.mean(np.array(samezies)))
            probs[exp] = exp_probs

        x = np.arange(self.n_experiments)
        f, ax = plt.subplots()
        for n,p in enumerate(probs.values()):
            xx = np.random.normal(n, 0.1, len(p))
            ax.bar(n, np.mean(p),  color=[.4, .4, .4], alpha=.5)
            ax.scatter(xx, p, s=90, alpha=.9)
        ax.axhline(.5, color='k', linewidth=.5)
        ax.set(title="$p(origin == escape)$", xticks=x, xticklabels=probs.keys())

    def plot_pR_individuals_by_exp(self):
        target = 'Right_Medium'
        escapes = {}
        experiments_pR, experiments_pRo = {}, {}
        for exp in sorted(self.experiments):         
            sessions = get_trials_by_exp(exp, 'true', ['session_uid'])
            experiment_pr, experiment_pro = [], []
            for uid in set(sessions):
                origin, escape = get_trials_by_exp_and_session(exp, uid, 'true', ['origin_arm', 'escape_arm'])
                pR = self.calc_arm_p(escape, target)
                pRo = self.calc_arm_p(origin, target)
                experiment_pr.append(pR)
                experiment_pro.append(pRo)

            experiments_pR[exp] = np.array(experiment_pr)
            experiments_pRo[exp] = np.array(experiment_pro)

        x = np.arange(self.n_experiments)
        f, ax = plt.subplots()
        for n, (pR, pRo) in enumerate(zip(experiments_pR.values(), experiments_pRo.values())):
            xx = np.random.normal(n, 0.1, len(pR))
            ax.bar(n, np.mean(pR),  color=[.4, .4, .4], alpha=.5)
            ax.bar(n, -np.mean(pRo),  color=[.2, .2, .2], alpha=.5)
            ax.scatter(xx, pR, s=90, alpha=.9)
            ax.scatter(xx, -pRo, s=90, alpha=.9)

        ax.axhline(.5, color='k', linewidth=.5)
        ax.axhline(-.5, color='k', linewidth=.5)
        ax.set(title="$p(R)$", xticks=x, xticklabels=experiments_pR.keys())

    def modelbased(self):
        exp = 'Model Based'
        sessions = get_trials_by_exp(exp, 'true', ['session_uid'])
        experiment_pr, experiment_pro = [], []
        arms = ['Left_Far', 'Centre', 'Right2']
        probs = {a:[] for a in arms}

        x = np.arange(3)
        f, axarr = plt.subplots(ncols=2)

        all_escapes = []
        mean_all_p = []
        for uid in sorted(set(sessions)):
            origin, escape = get_trials_by_exp_and_session(exp, uid, 'true', ['origin_arm', 'escape_arm'])

            all_escapes.extend(escape)
            p = [self.calc_arm_p(escape, a) for a in arms]
            mean_all_p.append(p)
            print(uid, len(escape), p)

            axarr[1].scatter(x, p, c='k', s=4*len(escape), alpha=.5)
            axarr[1].plot(x, p,  alpha=.5)

        mean_all_p = np.sum(np.vstack(mean_all_p), 0)/np.vstack(mean_all_p).shape[0]
        axarr[1].scatter(x, mean_all_p, c='r', s=50, alpha=1)

        all_probs = [self.calc_arm_p(all_escapes, a) for a in arms]
        colors = get_n_colors(len(set(sessions)))
        axarr[0].bar(x, all_probs, color=colors[:3])

        axarr[0].set(title="Model Based - Baseline", ylabel='p(R)', xticks=x, xticklabels=['Left', 'Centre', 'Right'])
        axarr[1].set(title="Model Based - Baseline", ylabel='p(R)', xticks=x, xticklabels=['Left', 'Centre', 'Right'])
        plt.show()


    

if __name__ == "__main__":
    p = Plotter()
    p.pr_sym_vs_asmy()


    # p.plot_pR_individuals_by_exp()
    # p.plot_p_same_origin_and_escape_by_exp()

    # p.plot_pR_by_exp()
    # p.modelbased()


    plt.show()