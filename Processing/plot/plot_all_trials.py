import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd

from database.NewTablesDefinitions import *
from database.database_fetch import *

from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame, get_arm_given_rois, convert_roi_id_to_tag
from Processing.tracking_stats.math_utils import get_roi_enters_exits, line_smoother, calc_distance_between_points_2d, remove_tracking_errors



class PlotAllTrials:
    def __init__(self, select_escapes=True):
        self.trials = AllTrials()
        self.save_fld = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\plots\\all_trials'

        if select_escapes:
            self.escapes = 'true'
        else:
            self.escapes = 'false'


    def plot_by_exp(self):
        experiments = set(AllTrials.fetch('experiment_name'))

        # Get all trials for each experiment, regardless of the mouse they belong to
        for exp in sorted(experiments):
            trials = (AllTrials & "experiment_name='{}'".format(exp) & "is_escape='{}'".format(self.escapes)).fetch('tracking_data')

            self.plot_trials(trials, exp)

    def plot_by_session(self):
        sessions = set(AllTrials.fetch('session_uid'))

        for uid in sessions:
            experiments, trials = (AllTrials & "session_uid='{}'".format(uid)).fetch('experiment_name', 'tracking_data')
            self.plot_trials(trials, experiments[0], uid)



    def plot_trials(self, trials, exp, label0=None):
        def plot_segment(ax, tracking, bp1, bp2, col):
            tot_frames = tracking[bp1][0].shape[0]
            sel_frames = np.linspace(0, tot_frames-1, tot_frames/4).astype(np.int16)
            xx = [tracking[bp1][0][sel_frames], tracking[bp2][0][sel_frames]]
            yy = [tracking[bp1][1][sel_frames], tracking[bp2][1][sel_frames]]
            if 'ear' in bp1 or 'ear' in bp2:
                col = 'r'

            ax.plot(xx, yy, color=col, alpha=.4)


        print('plotting...')
        maze_model = get_maze_template(exp=exp)

        f, axarr = plt.subplots(nrows=2, ncols=3, figsize=(16, 12))
        f2, tax = plt.subplots(figsize=(16, 12))
        axarr = axarr.flatten()
        for ax in axarr:
            ax.imshow(maze_model)
        tax.imshow(maze_model)

        bps = ['body', 'snout', 'left_ear', 'right_ear', 'neck', 'tail_base']
        for tr in trials:
            col = [np.random.uniform(.3, 1), np.random.uniform(.3, 1), np.random.random()]

            bps_x = [remove_tracking_errors(tr[:, :, i])[:, 0] for i,_ in enumerate(bps)]
            bps_y = [remove_tracking_errors(tr[:, :, i])[:, 1] for i,_ in enumerate(bps)]

            tracking = {bp:np.array([bps_x[i], bps_y[i]]).T for i,bp in enumerate(bps)}
            tot_frames = bps_x[0].shape[0]
            sel_frames = np.linspace(0, tot_frames-1, tot_frames/30).astype(np.int16)

            tracking_array = np.array([np.vstack(bps_x), np.vstack(bps_y)]).T

            for ax_n, ax in enumerate(axarr):
                poly_bp = ['left_ear','snout',  'right_ear', 'neck']
                body_poly_bp = ['left_ear', 'tail_base', 'right_ear', 'neck']
                colors=['r', 'g']
                for frame in np.arange(tot_frames):
                    if frame not in sel_frames:
                        fill = None
                        alpha=.1
                    else:
                        fill = True
                        alpha=1



                    for bbpp, color in zip([poly_bp, body_poly_bp], colors):
                        poly = np.hstack([np.vstack([tracking[bp][frame, 0] for bp in bbpp]),
                                        np.vstack([tracking[bp][frame, 1] for bp in bbpp])])

                        if frame == 0:
                            lw = 3
                            edgecolor = 'k'
                            fill=True
                            alpha=1
                        else:
                            edgecolor=color
                            lw=1
                            fill=None
                            alpha=.05

                        
                        axpoly = Polygon(poly, fill=fill, facecolor=color, edgecolor=edgecolor, lw=lw, alpha=alpha)
                        ax.add_patch(axpoly)
                        taxpoly = Polygon(poly, fill=fill, facecolor=color, edgecolor=edgecolor, lw=lw, alpha=alpha)
                        tax.add_patch(taxpoly)

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
        tax.set(title=savename, xlim=[400, 600], ylim=[50, 430], yticks=[], xticks=[])

        f.tight_layout()
        # f.savefig(os.path.join(self.save_fld, savename+'.svg'), format="svg")
        f.savefig(os.path.join(self.save_fld, savename+'.png'))
        f2.savefig(os.path.join(self.save_fld, savename+'_threat.png'))

        plt.close(f)
        plt.close(f2)



if __name__ == "__main__":
    plotter = PlotAllTrials(select_escapes=True)
    plotter.plot_by_exp()
    # plotter.plot_by_session()


    plt.show()

















