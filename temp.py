import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np  
import matplotlib.pyplot as plt
sys.path.append('./')   
from nptdms import TdmsFile
from database.dj_config import start_connection
from database.NewTablesDefinitions import *


def run():
    # Loop over each experiment
    experiments = Experiments()
    sessions = Sessions()
    stimuli = BehaviourStimuli()
    videofiles = VideoFiles()
    tracking = TrackingData()

    fetched_sessions = sessions.fetch(as_dict=True)
    fetched_stimuli = stimuli.fetch(as_dict=True)
    fetched_bodyparts = tracking.BodyPartData.fetch(as_dict=True)

    for exp in experiments.fetch(as_dict=True):
        f, ax = plt.subplots(facecolor=[.2, .2, .2])
        check = False
        # Loop over each session for each experiment
        for session in fetched_sessions:
            if session['experiment_name'] != exp['experiment_name']:
                continue
            
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
                ax.plot(stim_tracking[0]['tracking_data'][stim['stim_start']-60:stim['stim_start']+timewindow,0],
                        stim_tracking[0]['tracking_data'][stim['stim_start']-60:stim['stim_start']+timewindow,1],
                        color=[.8, .8, .8], alpha=.75)
        print('Experiment: ', exp['experiment_name'])
        if check:
            ax.set_ylim(ax.get_ylim()[::-1])
            ax.set(title =  exp['experiment_name'], facecolor=[.2, .2, .2])
    plt.show()


if __name__ == "__main__":
        run()


