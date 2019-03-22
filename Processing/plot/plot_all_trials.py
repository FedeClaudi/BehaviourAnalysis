import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.image as mpimg
import pandas as pd
import os
import seaborn as sns
import math

from database.NewTablesDefinitions import *
from database.database_fetch import *

from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame, get_arm_given_rois, convert_roi_id_to_tag
from Processing.tracking_stats.math_utils import *
from Utilities.video_and_plotting.video_editing import Editor

class PlotAllTrials:
    def __init__(self, select_escapes=True):
        self.trials = AllTrials()
        self.save_fld = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\plots\\all_trials'

        if select_escapes:
            self.escapes = 'true'
        else:
            self.escapes = 'false'

        self.to_fetch = ['experiment_name', 'tracking_data', 'fps', 'number_of_trials', 'trial_number', 
                         'recording_uid', 'stim_frame', 'escape_arm', 'origin_arm', 'is_escape', 'trial_id']


    """
    #####################################################################################################################################################################################################################################################
    #####################################################################################################################################################################################################################################################
    #####################################################################################################################################################################################################################################################
    """

    def plot_all_trials(self):
        experiments, trials, fps, number_of_trials, trial_number, rec_uid, stim_frames, escapes, origins, is_escape, trial_ids = \
                (AllTrials & "is_escape='{}'".format(self.escapes)).fetch(*self.to_fetch)

        self.plot_as_video(trial_ids, trials, 'ALL TRIALS', fps, rec_uid, stim_frames, escapes, origins,
                           None, number_of_trials, trial_number, savename=arm)

    def plot_by_arm(self):
        arms = set((AllTrials).fetch("escape_arm"))
        for arm in arms:
            experiments, trials, fps, number_of_trials, trial_number, rec_uid, stim_frames, escapes, origins, is_escape = \
                (AllTrials & "escape_arm='{}'".format(arm) & "is_escape='{}'".format(self.escapes)).fetch(*self.to_fetch)

            if not np.any(fps): fps = 30
            else: fps=fps[0]

            self.plot_as_video(trials, "allexp", fps, rec_uid, stim_frames, escapes, origins,
                                None, number_of_trials, trial_number, savename=arm)


    def plot_by_exp(self):
        experiments = set(AllTrials.fetch('experiment_name'))

        # Get all trials for each experiment, regardless of the mouse they belong to
        for exp in sorted(experiments):           
            experiments, trials, fps, number_of_trials, trial_number, rec_uid, stim_frames, escapes, origins, is_escape= \
                (AllTrials & "experiment_name='{}'".format(exp) & "is_escape='{}'".format(self.escapes))\
                            .fetch(*self.to_fetch)

            if not np.any(fps): continue
            self.plot_as_video(trials, exp, fps[0], rec_uid, stim_frames, escapes, origins, \
                        None, number_of_trials, trial_number)

    def plot_by_session(self, as_video=False):
        sessions = set(AllTrials.fetch('session_uid'))

        for uid in sessions:
            experiments, trials, fps, number_of_trials, trial_number, rec_uid, stim_frames, escapes, origins, is_escape = \
                (AllTrials & "session_uid='{}'".format(uid) & "is_escape='{}'".format(self.escapes)).fetch(*self.to_fetch)

            if not np.any(experiments): continue


            self.plot_as_video(trials, experiments[0], fps[0], rec_uid, stim_frames, escapes, origins, \
                                uid, number_of_trials[0], trial_number, is_escape=is_escape)

    def plot_by_feature(self, feature):
        # Get notes and sort them
        features = load_yaml('Processing\\trials_analysis\\trials_observations.yml')[feature]
        uids = np.array([u for u, n in features])
        trials_n = np.array([n for u,n in features])

        experiments, trials, fps, number_of_trials, trial_number, rec_uid, stim_frames = [],[],[],[],[],[],[],
        escapes, origins = [], []
        for uid, n in zip(uids, trials_n):

            exp, tr, fp, numtr, trnum, r, sfr, esc, ori = (AllTrials & "session_uid='{}'".format(uid) & "trial_number={}".format(n))\
                            .fetch(*self.to_fetch)

            if not np.any(exp): continue

            experiments.append(exp[0])
            trials.append(tr[0])
            fps.append(fp[0])
            number_of_trials.append(numtr[0])
            trial_number.append(trnum[0])
            rec_uid.append(r[0])
            stim_frames.append(sfr[0])
            escapes.append(esc)
            origins.append(ori)      

        self.plot_as_video(trials, experiments, 35, rec_uid, stim_frames, escapes, origins, \
                            uid, number_of_trials, trial_number, savename=feature)


    """
    #####################################################################################################################################################################################################################################################
    #####################################################################################################################################################################################################################################################
    #####################################################################################################################################################################################################################################################
    """



    def plot_as_video(self, trial_ids, trials, exp,  fps, rec_uids, stim_frames, escapes, origins, \
                        label0=None, number_of_trials = None, trial_number=None, savename=None, is_escape=None):
        def draw_contours(fr, cont, c1, c2):
            if c2 is not None:
                cv2.drawContours(fr, cont, -1, c2, 4)
            cv2.drawContours(fr, cont, -1, c1, -1)    

        # Get name of file to be saved and check if it excists already
        if savename is None:
            if label0 is not None:
                savename = str(label0) + '-' + exp
            else:
                savename = exp
        
        complete_name = os.path.join(self.save_fld, savename+'.mp4')
        if os.path.isfile(complete_name): return

        # Get maze model, idxs of bodypats for contours, location of cropping for threat pltform...
        if not isinstance(exp, list):
            maze_model = get_maze_template(exp=exp)
        else:
            maze_model = get_maze_template()

        bps = ['body', 'snout', 'left_ear', 'right_ear', 'neck', 'tail_base']
        body_idxs = [2, 1, 3, 5]
        head_idxs = [2, 1, 3, 0]
        threat_cropping = ((570, 800), (400, 600))

        # open openCV writer
        video_editor = Editor()
        writer = video_editor.open_cvwriter(complete_name,
                                            w = maze_model.shape[0]*2, h=maze_model.shape[1],
                                            framerate = fps, iscolor=True) 

        # loop over each trial
        stored_contours = []
        for n, (trid, tr, trn, rec_uid, stim_frame, escs, ors) in enumerate(zip(trial_ids, trials, trial_number, rec_uids, stim_frames, escapes, origins)): #, is_escape)):
            is_esc = ''
            # shift all tracking Y up by 10px to align better
            trc = tr.copy()
            trc[:, 1, :] = np.add(trc[:, 1, :], 10)
            tr = trc


            # cv2.namedWindow('frame',cv2.WINDOW_AUTOSIZE )
            tot_frames = tr.shape[0]

            # get tracking data for the different contours to draw
            selected_tracking = tr[:, :, body_idxs]
            head_tracking = tr[:, :, head_idxs]
            body_ellipse = tr[:, :, [0, 5]]

            # If we have trials from different experiments, fetch the correct maze model
            if isinstance(exp, list):
                maze_model=get_maze_template(exp=exp[n])

            # Make trial background [based on previous trials]            
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

            # open recorded video and move to stim start frame
            videopath = get_video_path_give_recuid(rec_uid)
            cap = cv2.VideoCapture(videopath)
            if not cap.isOpened():
                raise FileNotFoundError
            else:
                cap.set(1, stim_frame)

            # Compute body and head angles and IdPhi
            body_angle = calc_angle_between_vectors_of_points_2d(tr[:, :2, -1].T, tr[:, :2, 0].T)
            head_angle = calc_angle_between_vectors_of_points_2d(tr[:, :2, 0].T, tr[:, :2, 1].T)
            # IdPhi = calc_IdPhi(-body_angle[:t_exit[0]])

            trial_stored_contours = []

            prev_frame_bl = 0
            for frame in np.arange(tot_frames): # loop over each frame and draw
                background = trial_background.copy()

                # Get body ellipse
                try:
                    centre = (int(np.mean([body_ellipse[frame, 0, 0], body_ellipse[frame, 0, 1]])),
                                int(np.mean([body_ellipse[frame, 1, 0], body_ellipse[frame, 1, 1]])))
                except:
                    continue

                main_axis = int(calc_distance_between_points_2d(body_ellipse[frame, :2, 0], body_ellipse[frame, :2, 1]))
                if main_axis > 100: main_axis = 10
                elif main_axis < 15: main_axis = 15
                if prev_frame_bl:
                    if abs(prev_frame_bl - main_axis) > prev_frame_bl*3:
                        main_axis = prev_frame_bl
                    
                prev_frame_bl = main_axis
                
                min_axis = int(main_axis*.3)
                angle = angle_between_points_2d_clockwise(body_ellipse[frame, :2, 0], body_ellipse[frame, :2, 1])
                cv2.ellipse(background, centre, (min_axis,main_axis), angle-90, 0, 360, (0, 255, 0), -1)
                cv2.ellipse(background, centre, (min_axis,main_axis), angle-90, 0, 360, (50, 50, 50), 2)

                # Draw current trial head contours
                coords = selected_tracking[frame, :2, :].T.astype(np.int32)
                head = head_tracking[frame, :2, :].T.astype(np.int32)
                draw_contours(background, [head],  (0, 0, 255), (50, 50, 50))

                # flip frame Y
                background = np.array(background[::-1, :, :])

                # Title
                if isinstance(number_of_trials, list):
                    n_of_t = number_of_trials[n]
                else:
                    n_of_t = number_of_trials

                if isinstance(exp, list):
                    ttl = savename + ' - ' + exp[n]
                else:
                    ttl = savename

                cv2.putText(background, ttl + '- trial id: ' + str(trid), 
                            (int(maze_model.shape[1]/10), int(maze_model.shape[1]/10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)

                # Time elapsed
                elapsed = frame / fps
                cv2.putText(background, str(round(elapsed, 2)),
                            (int(maze_model.shape[0]*.85), int(maze_model.shape[1]*.8)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, 
                            (255, 255, 255), 2,cv2.LINE_AA)

                # Escape and Origin Arms
                if isinstance(ors, list) or isinstance(ors, np.ndarray): ors = ors[0]
                if isinstance(escs, list) or isinstance(escs, np.ndarray): escs=escs[0]
                cv2.putText(background, 'origin arm: '+ ors,
                            (int(maze_model.shape[0]*.1), int(maze_model.shape[1]*.15)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (255, 255, 255), 2,cv2.LINE_AA)
                cv2.putText(background, 'escape arm: '+ escs,
                            (int(maze_model.shape[0]*.1), int(maze_model.shape[1]*.2)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (255, 255, 255), 2,cv2.LINE_AA)

                cv2.putText(background, 'is escape: '+ is_esc,
                            (int(maze_model.shape[0]*.6), int(maze_model.shape[1]*.15)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (255, 255, 255), 2,cv2.LINE_AA)

                # Head and body angle
                cc = (int(maze_model.shape[0]*.66), int(maze_model.shape[1]*.88))
                cv2.circle(background, cc, 100, (255, 255, 255), 2)
                cv2.ellipse(background, cc, (100, 5), -body_angle[frame], -45, 45, (0, 255, 0), -1)
                cv2.ellipse(background, cc, (50, 10), -head_angle[frame], -45, 45, (0, 0, 255), -1)
                cv2.ellipse(background, cc, (100, 2), -head_angle[frame] - (-body_angle[frame]) - 90, -45, 45, (255, 255, 255), -1)

                # threat frame
                threat = background[threat_cropping[0][0]:threat_cropping[0][1],
                                    threat_cropping[1][0]:threat_cropping[1][1]]
                threat = cv2.resize(threat, (background.shape[1], background.shape[0]))

                # video frame
                ret, videoframe = cap.read()
                if not ret:
                    raise ValueError

                wh_ratio = videoframe.shape[0] / videoframe.shape[1] 
                height = 350
                width = int(250*wh_ratio)

                videoframe = cv2.resize(videoframe, (height, width))

                # Create the whole frame
                shape = background.shape
                whole_frame = np.zeros((shape[0], shape[1]*2, shape[2])).astype(np.uint8)
                whole_frame[:, :shape[0],:] = background
                whole_frame[:, shape[0]:,:] = threat
                whole_frame[shape[0]-width:, :height, :] = videoframe

                # Show and write
                # cv2.imshow("frame", whole_frame)
                # cv2.waitKey(1)
                writer.write(whole_frame)

                # Store contours points of this trials to use them as background for next
                trial_stored_contours.append(coords)

            stored_contours.append(trial_stored_contours)
        writer.release()



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

    """
    #####################################################################################################################################################################################################################################################
    #####################################################################################################################################################################################################################################################
    #####################################################################################################################################################################################################################################################
    """


if __name__ == "__main__":
    plotter = PlotAllTrials(select_escapes=True)
    # plotter.plot_by_exp()
    # plotter.plot_by_session(as_video=True)

    plotter.plot_by_arm()

    # features_keys = load_yaml('Processing\\trials_analysis\\trials_observations.yml').keys()
    # for feature in features_keys:
    #     plotter.plot_by_feature(feature)

    # plotter.visualise_plots()

    # plotter.plot_dPhi_by_exp()


    plt.show()

















