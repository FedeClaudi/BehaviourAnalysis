import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np  
import matplotlib.pyplot as plt
sys.path.append('./')   
from nptdms import TdmsFile
from database.dj_config import start_connection
from database.NewTablesDefinitions import *
import cv2

def run():
    def plotter(ax1, ax2, color, x, y, x_pad, y_pad):
        padded_x = np.add(x, x_pad)
        padded_y = np.add(y, y_pad)
        ax1.plot(padded_x, padded_y, color=color, alpha=.75)
        ax2.plot(padded_x, padded_y, color=color, alpha=.75)

    # Loop over each experiment
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

    f, axarr = plt.subplots(4, 3, facecolor=[.2, .2, .2])
    axarr = axarr.flatten()
    background = cv2.imread('Utilities\\video_and_plotting\\mazemodel.png')
    background = cv2.resize(background, (1000, 1000))
    #background = cv2.cv2.cvtColor(background,cv2.COLOR_RGB2GRAY)
    [ax.imshow(background) for ax in axarr]
    color=iter(plt.cm.rainbow(np.linspace(0,1,12)))
    for i, exp in enumerate(experiments.fetch(as_dict=True)):
        if 'yml' in exp['experiment_name']: continue
        check = False
        c=next(color)

        # Loop over each session for each experiment
        for session in fetched_sessions:
            if session['experiment_name'] != exp['experiment_name']:
                continue

            # Get X and Y offest
            matrix = [m for m in fetched_ccm if m['uid'] == session['uid']]
            if not matrix: continue
            else: matrix=matrix[0]
            x_offset, y_offset = matrix['side_pad'], matrix['top_pad']

            # Loop over each behaviour stimulus
            for stim in fetched_stimuli:
                if stim['uid'] != session['uid']:
                    continue
                stim_tracking = [t for t in fetched_bodyparts 
                                if t['recording_uid']==stim['recording_uid'] 
                                and t['bpname']=='body']
                if not stim_tracking: 
                    continue
                    mm = [t for t in tracking.BodyPartData.fetch(as_dict=True) 
                                if t['recording_uid']==stim['recording_uid']]
                    raise ValueError('no stims found: ', mm, stim)
                
                check = True
                timewindow = 400

                x = stim_tracking[0]['tracking_data'][stim['stim_start']-60:stim['stim_start']+timewindow,0]
                y = stim_tracking[0]['tracking_data'][stim['stim_start']-60:stim['stim_start']+timewindow,1]
                plotter(axarr[i], axarr[-1], c, x, y, x_offset, y_offset)


        print('Experiment: ', exp['experiment_name'])
        # axarr[i].set_ylim(axarr[i].get_ylim()[::-1])
        axarr[i].set(title =  exp['experiment_name'])

    # axarr[-1].set_ylim(axarr[-1].get_ylim()[::-1])
    axarr[-1].set(title = 'All experiments')
    plt.show()
    f.tight_layout()

if __name__ == "__main__":
        run()


