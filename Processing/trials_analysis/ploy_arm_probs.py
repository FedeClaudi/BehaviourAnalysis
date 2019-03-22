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


    def pr_sym_vs_asmy(self):
        def plot_two_dists_kde(d1, d2, title, l1=None, l2=None):
            colors = get_n_colors(6)
            f, ax = plt.subplots()

            c1, c2 = colors[2], colors[3]

            sns.kdeplot(d1, ax=ax, shade=True, color=c1, linewidth=2, alpha=.8, clip=[0, 1], label=l1)
            sns.kdeplot(d2, ax=ax, shade=True, color=c2, linewidth=2, alpha=.8, clip=[0, 1], label=l2)
            ax.set(title=title, xlim=[0, 1])
            ax.legend()
            f.savefig("D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Presentations\\ThesisCommitte\\plots\\{}.svg".format(title.strip().split('-')[0]), format="svg")

        bayes = Bayesian()


        # Get data
        asym_exps = ["PathInt2", "PathInt2-L"]
        sym_exps = ["Square Maze", "TwoAndahalf Maze"]

        asym = [arm for arms in [get_trials_by_exp(e, 'true', ['escape_arm']) for e in asym_exps] for arm in arms]
        sym = [arm for arms in [get_trials_by_exp(e, 'true', ['escape_arm']) for e in sym_exps] for arm in arms]

        asym_origins = [arm for arms in [get_trials_by_exp(e, 'true', ['origin_arm']) for e in asym_exps] for arm in arms]
        sym_origins = [arm for arms in [get_trials_by_exp(e, 'true', ['origin_arm']) for e in sym_exps] for arm in arms]

        asym_sessions = [s for sessions in [get_sessuids_given_experiment(e) for e in asym_exps] for s in sessions]
        sym_sessions = [s for sessions in [get_sessuids_given_experiment(e) for e in sym_exps] for s in sessions]

        # Get p(R) for all trials
        p_asym = self.calc_arm_p(asym, "Right_Medium")
        p_sym = self.calc_arm_p(sym, "Right_Medium")

        """
            LOOK AT EFFECT OF ARM OF ORIGIN
        """
        asym_r_ori = [e for o,e in zip(asym_origins, asym) if 'Right' in o]
        sym_r_ori = [e for o,e in zip(sym_origins, sym) if 'Right' in o]

        asym_l_ori = [e for o,e in zip(asym_origins, asym) if 'Left' in o]
        sym_l_ori = [e for o,e in zip(sym_origins, sym) if 'Left' in o]

        # DO SOME MODELLING
        if 1 == 1:
            asym_r_ori_int = [1 if 'Right' in e else 0 for e in asym_r_ori]
            asym_l_ori_int = [1 if 'Right' in e else 0 for e in asym_l_ori]
            trace, D, dp, t, tp = bayes.model_two_distributions(asym_r_ori_int, asym_l_ori_int)
            plot_two_dists_kde(trace['p_d1'].values, trace['p_d2'].values, 'ASYM $p(R)$ given origin - D:{},p{}'.format(t, tp), 'R origin', 'L origin')
            

            sym_r_ori_int = [1 if 'Right' in e else 0 for e in sym_r_ori]
            sym_l_ori_int = [1 if 'Right' in e else 0 for e in sym_l_ori]
            trace,  D, dp, t, tp = bayes.model_two_distributions(sym_r_ori_int, sym_l_ori_int)
            plot_two_dists_kde(trace['p_d1'].values, trace['p_d2'].values, 'SYM $p(R)$ given origin - D:{},p{}'.format(t, tp), 'R origin', 'L origin')

        """
        p_asym_r_lori = self.calc_arm_p(asym_l_ori, "Right_Medium")
        p_sym_r_lori = self.calc_arm_p(sym_l_ori, "Right_Medium")
        p_asym_r_rori = self.calc_arm_p(asym_r_ori, "Right_Medium")
        p_sym_r_rori = self.calc_arm_p(sym_r_ori, "Right_Medium")

        # Stuf for plottings
        labels = ["asymm.", "symm."]
        xx = np.arange(len(labels))
        xx2 = [0,1,3,4]
        yy = np.linspace(0, 10, 11)/10
        colors = get_n_colors(len(labels))

        # Plot the probability of going right
        f, ax = plt.subplots(figsize=(8, 12))

        ax.bar(xx, [p_asym, p_sym], color=colors)
        ax.axhline(.5, linestyle=":")
        ax.set(title="p(R) for all trials", xticks=xx, xticklabels=labels, yticks=yy, ylabel="p(R)", ylim=[0, 1])

        f.savefig("D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Presentations\\ThesisCommitte\\plots\\pR_asym_sym.svg", format="svg")

        # Plot the probability of going left given a left origin
        f, ax = plt.subplots(figsize=(8, 12))

        ax.bar(xx2, [p_asym_r_lori, p_asym_r_rori, p_sym_r_lori, p_sym_r_rori], color=colors)

        ax.axhline(.5, linestyle=":")
        ax.set(title="p(R) given R origin and given L origin", xticks=xx, xticklabels=labels, yticks=yy, ylabel="p(R)", ylim=[0, 1])

        f.savefig("D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Presentations\\ThesisCommitte\\plots\\pR_asym_sym_given_lori.svg", format="svg")
        """

        """
            LOOK AT THE EFFECT OF X POSITION
        """

        # Plot the probs of escaping left and right based on the position at stim onset
        asym_tracking = [arm for arms in [get_trials_by_exp(e, 'true', ['tracking_data']) for e in asym_exps] for arm in arms]
        sym_tracking = [arm for arms in [get_trials_by_exp(e, 'true', ['tracking_data']) for e in sym_exps] for arm in arms]

        asym_position_onset = [1 if 480 > tr[0, 0, 0] else 2 if 520 < tr[0, 0, 0] else 0 for tr in asym_tracking ]
        sym_position_onset = [1 if 480 > tr[0, 0, 0] else 2 if 520 < tr[0, 0, 0] else 0 for tr in sym_tracking ]

        asym_position_onset_pos = [tr[0, :2, 0] for tr in asym_tracking if (480 > tr[0, 0, 0] or 520 < tr[0, 0, 0])]
        sym_position_onset_pos = [tr[0, :2, 0] for tr in sym_tracking if (480 > tr[0, 0, 0] or 520 < tr[0, 0, 0])]

        asym_l_pos, asym_r_pos = [e for i,e in enumerate(asym) if asym_position_onset[i]==1], [e for i,e in enumerate(asym) if asym_position_onset[i]==2]
        sym_l_pos, sym_r_pos = [e for i,e in enumerate(sym) if sym_position_onset[i]==1], [e for i,e in enumerate(sym) if sym_position_onset[i]==2]

        # asym_escape_all_position = [e for i,e in enumerate(asym) if asym_position_onset[i]!=0]
        # sym_escape_all_position = [e for i,e in enumerate(sym) if sym_position_onset[i]!=0]

        # asym_p_r_lpos, asym_p_r_rpos = self.calc_arm_p(asym_l_pos, 'Right_Medium'), self.calc_arm_p(asym_r_pos, 'Right_Medium')
        # sym_p_r_lpos, sym_p_r_rpos = self.calc_arm_p(sym_l_pos, 'Right_Medium'), self.calc_arm_p(sym_r_pos, 'Right_Medium')

        # do some MODELLING
        if 1 == 1:
            asym_l_pos_int, asym_r_pos_int = [1 if 'Right' in e else 0 for e in asym_l_pos], [1 if 'Right' in e else 0 for e in asym_r_pos]
            sym_l_pos_int, sym_r_pos_int = [1 if 'Right' in e else 0 for e in sym_l_pos], [1 if 'Right' in e else 0 for e in sym_r_pos]

            trace,  D, dp, t, tp = bayes.model_two_distributions(asym_r_pos_int, asym_l_pos_int)
            plot_two_dists_kde(trace['p_d1'].values, trace['p_d2'].values, 'ASYM $p(R)$ given X position - D:{},p{}'.format(t, tp), 'R pos', 'L pos')
            trace,  D, dp, t, tp = bayes.model_two_distributions(sym_r_pos_int, sym_l_pos_int)
            plot_two_dists_kde(trace['p_d1'].values, trace['p_d2'].values, 'SYM $p(R)$ given X position - D:{},p{}'.format(t, tp), 'R pos', 'L pos')

        """
            LOOK AT THE EFFECT OF ORIENTATION
        """

        asym_body_pos, asym_tail_pos = np.vstack([tr[0, :2, 0] for tr in asym_tracking ]), np.vstack([tr[0, :2, -1] for tr in asym_tracking ])
        sym_body_pos, sym_tail_pos = np.vstack([tr[0, :2, 0] for tr in sym_tracking ]), np.vstack([tr[0, :2, -1] for tr in sym_tracking ])

        asym_orient = calc_angle_between_vectors_of_points_2d(asym_body_pos.T, asym_tail_pos.T)
        sym_orient = calc_angle_between_vectors_of_points_2d(sym_body_pos.T, sym_tail_pos.T)

        asym_rorient, asym_lorient = [e for i,e in enumerate(asym) if asym_orient[i] <= 90-22.5], [e for i,e in enumerate(asym) if 180 >= asym_orient[i] >= 90+22.5]
        sym_rorient, sym_lorient = [e for i,e in enumerate(sym) if sym_orient[i] <= 90-22.5], [e for i,e in enumerate(sym) if 180 >= sym_orient[i] >= 90+22.5]
        
        asym_rorient_int, asym_lorient_int= [1 if 'Right' in e else 0 for e in asym_rorient], [1 if 'Right' in e else 0 for e in asym_lorient]
        sym_rorient_int, sym_lorient_int= [1 if 'Right' in e else 0 for e in sym_rorient], [1 if 'Right' in e else 0 for e in sym_lorient]

        trace, D, dp, t, tp = bayes.model_two_distributions(asym_rorient_int, asym_lorient_int)
        plot_two_dists_kde(trace['p_d1'].values, trace['p_d2'].values, 'ASYM $p(R)$ given Orientation - D:{},p{}'.format(t, tp), 'R oriented', 'L oriented')
        trace, D, dp, t, tp = bayes.model_two_distributions(sym_rorient_int, sym_lorient_int)
        plot_two_dists_kde(trace['p_d1'].values, trace['p_d2'].values, 'SYM $p(R)$ given Orientation - D:{},p{}'.format(t, tp), 'R oriented', 'L oriented')

        plt.show()

        """
        f, ax = plt.subplots(figsize=(8, 12))
        ax.bar(xx2, [asym_p_r_lpos, asym_p_r_rpos, sym_p_r_lpos, sym_p_r_rpos], color=colors)
        ax.set(title="P(R) given position on the left vs given position on the right", xticks=xx, xticklabels=labels, yticks=yy, ylabel="p(R)", ylim=[0, 1])
        f.savefig("D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Presentations\\ThesisCommitte\\plots\\pR_asym_sym_given_pos.svg", format="svg")
            
        # Plot the probs of escaping left and right based on the position at stim onset
        asym_body_pos, asym_tail_pos = np.vstack([tr[0, :2, 0] for tr in asym_tracking ]), np.vstack([tr[0, :2, -1] for tr in asym_tracking ])
        sym_body_pos, sym_tail_pos = np.vstack([tr[0, :2, 0] for tr in sym_tracking ]), np.vstack([tr[0, :2, -1] for tr in sym_tracking ])

        asym_orient = calc_angle_between_vectors_of_points_2d(asym_body_pos.T, asym_tail_pos.T)
        sym_orient = calc_angle_between_vectors_of_points_2d(sym_body_pos.T, sym_tail_pos.T)

        asym_rorient, asym_lorient = [e for i,e in enumerate(asym) if asym_orient[i] <= 90-22.5], [e for i,e in enumerate(asym) if asym_orient[i] >= 90+22.5]
        sym_rorient, sym_lorient = [e for i,e in enumerate(sym) if sym_orient[i] <= 90-22.5], [e for i,e in enumerate(sym) if sym_orient[i] >= 90+22.5]

        asym_pr_orientation = [self.calc_arm_p(asym_lorient, 'Right_Medium'), self.calc_arm_p(asym_rorient, 'Right_Medium')]
        sym_pr_orientation = [self.calc_arm_p(sym_lorient, 'Right_Medium'), self.calc_arm_p(sym_rorient, 'Right_Medium')]

        f, ax = plt.subplots(figsize=(8, 12))
        ax.bar(xx2, [asym_pr_orientation[0], asym_pr_orientation[1], sym_pr_orientation[0], sym_pr_orientation[1]], color=colors)

        ax.set(title = "P(R) given orientation", xticks=xx, xticklabels=labels, yticks=yy, ylabel="p(R)", ylim=[0, 1])
        f.savefig("D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Presentations\\ThesisCommitte\\plots\\pR_asym_sym_given_orientation.svg", format="svg")
        """


if __name__ == "__main__":
    p = Plotter()
    # p.pr_sym_vs_asmy()
    # p.plot_pR_individuals_by_exp()
    # p.plot_p_same_origin_and_escape_by_exp()

    # p.plot_pR_by_exp()
    p.modelbased()


    plt.show()