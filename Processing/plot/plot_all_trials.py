import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.image as mpimg
import pandas as pd
import os

from database.NewTablesDefinitions import *
from database.database_fetch import *

from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame, get_arm_given_rois, convert_roi_id_to_tag
from Processing.tracking_stats.math_utils import *
from Utilities.video_and_plotting.video_editing import Editor

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

    def plot_by_session(self, as_video=False):
        sessions = set(AllTrials.fetch('session_uid'))

        for uid in sessions:
            experiments, trials = (AllTrials & "session_uid='{}'".format(uid) & "is_escape='{}'".format(self.escapes)).fetch('experiment_name', 'tracking_data')

            if not np.any(experiments): continue

            if not as_video:
                self.plot_trials(trials, experiments[0], uid)
            else:
                self.plot_as_video(trials, experiments[0], uid)



    def plot_trials(self, trials, exp, label0=None, show=True):
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
        f2, tax = plt.subplots(figsize=(12, 16))
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
        
        if not show:
            f.savefig(os.path.join(self.save_fld, savename+'.png'))
            f2.savefig(os.path.join(self.save_fld, savename+'_threat.png'))
            plt.close(f)
            plt.close(f2)
        else:
            plt.show()

    def plot_as_video(self, trials, exp,  label0=None, ):
        def draw_contours(fr, cont, c1, c2):
            if c2 is not None:
                cv2.drawContours(fr, cont, -1, c2, 4)
            cv2.drawContours(fr, cont, -1, c1, -1)
            

        video_editor = Editor()

        if label0 is not None:
            savename = str(label0) + '-' + exp
        else:
            savename = exp

        complete_name = os.path.join(self.save_fld, savename+'.mp4')
        if os.path.isfile(complete_name): return

        maze_model = get_maze_template(exp=exp)

        bps = ['body', 'snout', 'left_ear', 'right_ear', 'neck', 'tail_base']
        correct_idxs = [2, 1, 3, 5]
        head_idxs = [2, 1, 3, 0]

        stored_contours = []


        threat_cropping = ((570, 800), (400, 600))
        border_size = 0
        writer = video_editor.open_cvwriter(complete_name,
                                            w = maze_model.shape[0]*2+border_size*2, h=maze_model.shape[1]+border_size*2,
                                            framerate = 30, iscolor=True) 


        for n, tr in enumerate(trials):

            fps = 30 # ! fix this


            cv2.namedWindow('frame',cv2.WINDOW_AUTOSIZE )
            
            tot_frames = tr.shape[0]

            selected_tracking = tr[:, :, correct_idxs]
            head_tracking = tr[:, :, head_idxs]
            body_ellipse = tr[:, :, [0, 5]]

            # Make trial background            
            if n == 0:
                trial_background = maze_model.copy()
            else:
                trial_background = maze_model.copy()
                if stored_contours:

                    for past_trial in stored_contours:
                        mask = np.uint8(np.ones(trial_background.shape) * 0)
                        draw_contours(mask, past_trial,  (255, 255, 255), None)
                        mask = mask.astype(bool)
                        trial_background[mask] = trial_background[mask] * .8

            trial_stored_contours = []
            for frame in np.arange(tot_frames):
                background = trial_background.copy()

                # Get body ellipse
                centre = (int(np.mean([body_ellipse[frame, 0, 0], body_ellipse[frame, 0, 1]])),
                            int(np.mean([body_ellipse[frame, 1, 0], body_ellipse[frame, 1, 1]])))

                main_axis = int(calc_distance_between_points_2d(body_ellipse[frame, :2, 0], body_ellipse[frame, :2, 1]))
                min_axis = int(main_axis*.3)
                
                angle = angle_between_points_2d_clockwise(body_ellipse[frame, :2, 0], body_ellipse[frame, :2, 1])
                cv2.ellipse(background, centre, (min_axis,main_axis), angle-90, 0, 360, (0, 255, 0), -1)
                cv2.ellipse(background, centre, (min_axis,main_axis), angle-90, 0, 360, (50, 50, 50), 2)

                # Draw current trial contorus
                coords = selected_tracking[frame, :2, :].T.astype(np.int32)
                head = head_tracking[frame, :2, :].T.astype(np.int32)
                # draw_contours(background, [coords],  (0, 255, 0), (50, 50, 50))
                draw_contours(background, [head],  (0, 0, 255), (50, 50, 50))

                # Add border to background
                if frame < 9 * fps:  # ! fix stim duration
                    border_color = (255, 0, 0)
                else:
                    border_color = (10, 10, 10)

                # # flip Y
                background = background[::-1, :, :]
                background= cv2.copyMakeBorder(background, 
                                                border_size, border_size, border_size, border_size, 
                                                cv2.BORDER_CONSTANT, value=border_color)

                # Title
                cv2.putText(background, savename + '- trial ' + str(n+1),                         
                            (int(maze_model.shape[0]/10), int(maze_model.shape[1]/10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (255, 255, 255), 2,cv2.LINE_AA)

                # Time elapsed
                elapsed = frame / fps
                cv2.putText(background, str(round(elapsed, 2)),
                            (int(maze_model.shape[0]*.75), int(maze_model.shape[1]*.9)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (255, 255, 255), 2,cv2.LINE_AA)

                # threat frame
                threat = background[threat_cropping[0][0]:threat_cropping[0][1],
                                    threat_cropping[1][0]:threat_cropping[1][1]]
                threat = cv2.resize(threat, (background.shape[1], background.shape[0]))

                shape = background.shape
                whole_frame = np.zeros((shape[0], shape[1]*2, shape[2])).astype(np.uint8)
                whole_frame[:, :shape[0],:] = background
                whole_frame[:, shape[0]:,:] = threat

                # Show and write


                cv2.imshow("frame", whole_frame)
                cv2.waitKey(1)

                writer.write(whole_frame)


                # Store contours points of previous trials
                trial_stored_contours.append(coords)


            stored_contours.append(trial_stored_contours)
        
        
        writer.release()




        



    def visualise_plots(self):
        figs = [os.path.join(self.save_fld, f) for f in os.listdir(self.save_fld) if not "_threat" in f]
        threat_figs = [os.path.join(self.save_fld, f) for f in os.listdir(self.save_fld) if "_threat" in f]

        selected = 0
        while True:
            f, ax = plt.subplots(figsize=(12, 8))
            f2, ax2 = plt.subplots(figsize=(8, 12))

            ax.imshow(mpimg.imread(figs[selected]))
            ax2.imshow(mpimg.imread(threat_figs[selected]))

            f.tight_layout()
            f2.tight_layout()

            plt.show()

            pn = input("prev next quit")
            if pn == "p":
                selected -= 1
            elif pn == "n":
                selected += 1
            else:
                break






if __name__ == "__main__":
    plotter = PlotAllTrials(select_escapes=True)
    # plotter.plot_by_exp()
    plotter.plot_by_session(as_video=True)
    # plotter.visualise_plots()

    


    plt.show()

















