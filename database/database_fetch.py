import sys
sys.path.append('./')
from database.NewTablesDefinitions import *
import cv2


"""Bunch of functions to facilitate retrieving filtered data from the database tables
"""


def get_recordings_given_sessuid(uid, recordings = None, as_dict=True):
    if recordings is not None:
        return recordings.loc[recordings['uid'] == uid]
    else: return (Recordings & "uid='{}'".format(uid)).fetch(as_dict=as_dict)

def get_stimuli_given_sessuid(uid, stimuli=None):
    if stimuli is not None:
        if 'stim_duration' in stimuli.columns:
            return stimuli.loc[(stimuli['uid']==uid)&(stimuli['stim_duration'] != -1)]
        elif 'duration' in stimuli.columns:
            return stimuli.loc[(stimuli['uid']==uid)&(stimuli['duration'] != -1)]
        else: raise ValueError
    else:
        # Check the software of that session
        software = (Recordings & "uid='{}'".format(uid)).fetch("software")
        if software == "behaviour":
            return (BehaviourStimuli & "uid='{}'".format(uid) & "stim_duration != 1").fetch()
        else:
            return (BehaviourStimuli & "uid='{}'".format(uid) & "duration != 1").fetch()

def get_stimuli_give_recuid(rec_uid):
    software = (Recordings & "recording_uid='{}'".format(rec_uid)).fetch("software")
    if software == "behaviour":
        return (BehaviourStimuli & "recording_uid='{}'".format(rec_uid) & "stim_duration != 1").fetch(as_dict=True)
    else:
        return (MantisStimuli & "recording_uid='{}'".format(rec_uid) & "duration > -1").fetch(as_dict=True)

def get_tracking_given_bp(bp):
    fetched = pd.DataFrame((TrackingData.BodyPartData & 'bpname = "{}"'.format(bp)).fetch())
    return fetched.loc[fetched['bpname'] == bp]

def get_tracking_given_recuid(ruid, tracking):
    return tracking.loc[tracking['recording_uid']==ruid]

def get_tracking_given_recuid_and_bp(recuid, bp):
    fetched = pd.DataFrame((TrackingData.BodyPartData & 'bpname = "{}"'.format(bp) & 'recording_uid = "{}"'.format(recuid)).fetch())
    return fetched

def get_videometadata_given_recuid(ruid, videometadata):
    return videometadata.loc[videometadata['recording_uid'] == ruid]

def get_videometadata_given_sessuid(uid, videometadata):
    return videometadata.loc[videometadata['uid'] == uid]

def get_sessuid_given_recuid(recuid, sessions):
    r = recuid.split('_')
    session_name = r[0]+'_'+r[1]
    return sessions.loc[sessions['session_name']==session_name], session_name

def get_recs_given_sessuid(uid, recordings=None):
    if recordings is not None:
        return recordings.loc[recordings['uid'] == int(uid)]
    else:
        recs = pd.DataFrame((Recordings & "uid='{}'".format(uid)).fetch())
        return recs

def get_sessuid_given_sessname(name):
    return (Sessions & "session_name = '{}'".format(name)).fetch('uid')

def get_videometadata_given_recuid(rec, just_fps=True):
    if not just_fps:
        return (VideoFiles.Metadata & "recording_uid='{}'".format(rec)).fetch()
    else:
        return (VideoFiles.Metadata & "recording_uid='{}'".format(rec)).fetch("fps")

def get_exp_given_sessname(name):
    return (Sessions & "session_name='{}'".format(name)).fetch("experiment_name")







def get_maze_template(exp=None):
    if exp is None:
        maze_model = cv2.imread('Utilities\\Maze_templates\\mazemodel.png')
    else:
        exp = exp.lower()
        if 'pathint2' in exp:
            maze_model = cv2.imread('Utilities\\Maze_templates\\PathInt2.png')
        elif 'pathint' in exp:
            maze_model = cv2.imread('Utilities\\Maze_templates\\PathInt.png')
        elif 'square' in exp:
            maze_model = cv2.imread('Utilities\\Maze_templates\\Square Maze.png')
        elif 'twoandahalf' in exp:
            maze_model = cv2.imread('Utilities\\Maze_templates\\TwoAndahalf Maze.png')
        elif 'flipflop' in exp:
            maze_model = cv2.imread('Utilities\\Maze_templates\\FlipFlop Maze.png')
        elif 'twoarmslong' in exp:
            maze_model = cv2.imread('Utilities\\Maze_templates\\TwoArmsLong Maze.png')
        elif 'fourarms' in exp:
            maze_model = cv2.imread('Utilities\\Maze_templates\\FourArms Maze.png')
        else:
            maze_model = cv2.imread('Utilities\\Maze_templates\\mazemodel.png')
        
    maze_model = cv2.resize(maze_model, (1000, 1000))

    return maze_model
