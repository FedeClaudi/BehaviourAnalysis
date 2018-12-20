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

    for exp in experiments.fetch(as_dit=True):
        f, ax = plt.subplots()
        # Loop over each session for each experiment
        for session in sessions.fetch(as_dict=True):
            if session['experiment_name'] != exp['experiment_name']:
                continue
            
            # Loop over each behaviour stimulus
            for stim in stimuli.fetch(as_dict=True):
                if stim['uid'] != session['uid']:
                    continue
                
                stim_tracking = [t for t in tracking.BodyPartData.fetch(as_dict=True) 
                                if t['recording_uid']==stim['recording_uid'] 
                                and t['bpname']=='body'][0]
                
                timewindow = 400
                ax.plot(stim_tracking['tracking_data'][stim['stim_start']-60:stim['stim_start']+timewindow,0],
                        'k', alpha=.75)
        plt.show()


if __name__ == "__main__":
        run()


