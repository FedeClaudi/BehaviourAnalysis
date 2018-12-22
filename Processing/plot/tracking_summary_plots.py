import cv2

from nptdms import TdmsFile
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('./')

from database.NewTablesDefinitions import *
from database.dj_config import start_connection

def plot_all_trials():
    def plotter(ax1, ax2, color, x, y):
        ax1.plot(x, y, color=color, alpha=.75)
        ax2.plot(x, y, color=color, alpha=.75)

    # Get data from database
    experiments = Experiments()
    sessions = Sessions()
    stimuli = BehaviourStimuli()
    videofiles = VideoFiles()
    tracking = TrackingData()
    ccm = CommonCoordinateMatrices()

    print('fetching')
    fetched_sessions = sessions.fetch(as_dict=True)
    fetched_stimuli = stimuli.fetch(as_dict=True)
    fetched_bodyparts = tracking.BodyPartData.fetch(as_dict=True)
    fetched_ccm = ccm.fetch(as_dict=True)
    print('plotting')

    # Create figures
    axarr = []
    for i in range(12):
        f, ax = plt.subplots(facecolor=[.2, .2, .2])
        axarr.append(ax)
    f2, axall = plt.subplots(facecolor=[.2, .2, .2])

    # Plot the std maze model on each axis
    background = cv2.imread('Utilities\\video_and_plotting\\mazemodel.png')
    background = cv2.resize(background, (1000, 1000))
    background = np.rot90(background, 2)
    [ax.imshow(background) for ax in axarr]
    axall.imshow(background)

    # Loop over each experiment, get data and plot
    color = iter(plt.cm.rainbow(np.linspace(0, 1, 12)))
    for i, exp in enumerate(experiments.fetch(as_dict=True)):
        if 'yml' in exp['experiment_name']:
            continue
        c = next(color)

        # Loop over each session for each experiment
        for session in fetched_sessions:
            if session['experiment_name'] != exp['experiment_name']:
                continue

            # Get X and Y offest
            matrix = [m for m in fetched_ccm if m['uid'] == session['uid']]
            if not matrix:
                continue
            else:
                matrix = matrix[0]

            # Loop over each behaviour stimulus
            for stim in fetched_stimuli:
                if stim['uid'] != session['uid']:
                    continue
                stim_tracking = [t for t in fetched_bodyparts
                                 if t['recording_uid'] == stim['recording_uid']
                                 and t['bpname'] == 'body']
                if not stim_tracking:
                    continue
                    mm = [t for t in tracking.BodyPartData.fetch(as_dict=True)
                          if t['recording_uid'] == stim['recording_uid']]
                    raise ValueError('no stims found: ', mm, stim)

                timewindow = 400
                x = stim_tracking[0]['tracking_data'][stim['stim_start'] -
                                                      60:stim['stim_start']+timewindow, 0]
                y = stim_tracking[0]['tracking_data'][stim['stim_start'] -
                                                      60:stim['stim_start']+timewindow, 1]
                plotter(axarr[i], axall, c, x, y)

        print('Experiment: ', exp['experiment_name'])
        axarr[i].set(title=exp['experiment_name'])

    axall.set(title='All experiments')
    plt.show()
    f.tight_layout()


if __name__ == "__main__":
        plot_all_trials()
