import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from database.NewTablesDefinitions import *
from database.database_fetch import *

from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame, get_arm_given_rois, convert_roi_id_to_tag
from Processing.tracking_stats.math_utils import get_roi_enters_exits, line_smoother, calc_distance_between_points_2d



class PlotAllTrials:
    def __init__(self):
        self.trials = AllTrials()
        self.save_fld = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\plots\\all_trials'


    def plot_by_exp(self):
        experiments = set(AllTrials.fetch('experiment_name'))

        # Get all trials for each experiment, regardless of the mouse they belong to
        for exp in experiments:
            trials = (AllTrials & "experiment_name='{}'".format(exp)).fetch('tracking_data')

            self.plot_trials(trials, exp)

    def plot_by_session(self):
        sessions = set(AllTrials.fetch('session_uid'))

        for uid in sessions:
            experiments, trials = (AllTrials & "session_uid='{}'".format(uid)).fetch('experiment_name', 'tracking_data')
            self.plot_trials(trials, experiments[0], uid)



    def plot_trials(self, trials, exp, label0=None):
        print('plotting...')
        maze_model = get_maze_template(exp=exp)

        f, axarr = plt.subplots(nrows=2, ncols=3, figsize=(16, 12))
        axarr = axarr.flatten()
        for ax in axarr:
            ax.imshow(maze_model)

        for tr in trials:
            col = [np.random.uniform(.3, 1), np.random.uniform(.3, 1), np.random.random()]

            xx = [tr[:, 0, 0], tr[:, 0, 1]]
            yy = [tr[:, 1, 0], tr[:, 1, 1]]

            for ax_n, ax in enumerate(axarr):
                ax.scatter(tr[:, 0, 1], tr[:, 1, 1], alpha=.5, s=15, c=col)

                if ax_n > 2:
                    ax.plot(xx, yy, alpha=.5, linewidth=3, color=col)

        axarr[0].cla()
        axarr[2].cla()

        if label0 is not None:
            axarr[0].set(title=label0)
            savename = str(label0) + '-' + exp
        else:
            savename = exp

        axarr[1].set(title=exp, xlim=[0, 1000], ylim=[0, 1000], yticks=[], xticks=[])
        axarr[3].set(title='Left - mid platf', xlim=[200, 400], ylim=[400, 630], yticks=[], xticks=[])
        axarr[4].set(title='Threat', xlim=[400, 600], ylim=[50, 430], yticks=[], xticks=[])
        axarr[5].set(title='Right - mid platf', xlim=[600, 800], ylim=[400, 630], yticks=[], xticks=[])


        f.tight_layout()
        f.savefig(os.path.join(self.save_fld, savename+'.svg'), format="svg")
        f.savefig(os.path.join(self.save_fld, savename+'.png'))



if __name__ == "__main__":
    plotter = PlotAllTrials()
    plotter.plot_by_exp()
    # plotter.plot_by_session()


    plt.show()

















