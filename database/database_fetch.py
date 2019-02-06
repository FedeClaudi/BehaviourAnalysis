import sys
sys.path.append('./')
from database.NewTablesDefinitions import *


"""Bunch of functions to facilitate retrieving filtered data from the database tables
"""


def get_recordings_given_sessuid(uid, recordings):
    return recordings.loc[recordings['uid'] == uid]

def get_stimuli_given_sessuid(uid, stimuli):
    if 'stim_duration' in stimuli.columns:
        return stimuli.loc[(stimuli['uid']==uid)&(stimuli['stim_duration'] != -1)]
    elif 'duration' in stimuli.columns:
        return stimuli.loc[(stimuli['uid']==uid)&(stimuli['duration'] != -1)]
    else: raise ValueError

def get_tracking_given_recuid(ruid, tracking):
    return tracking.loc[tracking['recording_uid']==ruid]

def get_videometadata_given_recuid(ruid, videometadata):
    return videometadata.loc[videometadata['recording_uid'] == ruid]

def get_videometadata_given_sessuid(uid, videometadata):
    return videometadata.loc[videometadata['uid'] == uid]