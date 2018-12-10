from Utilities.video_and_plotting.commoncoordinatebehaviour import run as get_matrix
import sys
sys.path.append('./')

import datajoint as dj
from database.dj_config import start_connection
from nptdms import TdmsFile
import pandas as pd
import os
from collections import namedtuple
import numpy as np
import cv2
import warnings
from Utilities.file_io.files_load_save import load_yaml


from database.NewTablesDefinitions import *

""" 
    Collection of functions used to populate the dj.Import and dj.Compute
    tables defined in NewTablesDefinitions.py

    CommonCoordinateMatrices    ok
    Templates                   ok
    Recordings
    VideoFiles
    VideoMetadata
    ConvertedVideoFiles
    PoseFiles
    MetadataFiles
    AnalogInputs
    Stimuli
    TrackingData
"""

class ToolBox:
    def __init__(self):
        # Load paths to data folders
        self.paths = load_yaml('paths.yml')
        self.raw_video_folder = os.path.join(
            self.paths['raw_data_folder'], self.paths['raw_video_folder'])
        self.raw_metadata_folder = os.path.join(
            self.paths['raw_data_folder'], self.paths['raw_metadata_folder'])
        self.tracked_data_folder = self.paths['tracked_data_folder']
        
    def get_behaviour_recording_files(self):
        pass



def make_commoncoordinatematrices_table(key):
    """make_commoncoordinatematrices_table [Allows user to align frame to model
    and stores transform matrix for future use]
    
    Arguments:
        key {[dict]} -- [key attribute of table autopopulate method from dj]
    """

    # Get the maze model template
    maze_model = cv2.imread('Utilities\\video_and_plotting\\mazemodel.png')
    maze_model = cv2.resize(maze_model, (1000, 1000))
    maze_model = cv2.cv2.cvtColor(maze_model, cv2.COLOR_RGB2GRAY)

    # Get path to video of first recording
    rec = [r for r in VideoFiles if r['session_name']
            == key['session_name'] and r['camera_name']=='overview']

    if not rec:
        raise ValueError(
            'Did not find recording while populating CCM table. Populate recordings first! Session: ', key['session_name'])
    else:
        videopath = rec[0]

    # Apply the transorm [Call function that prepares data to feed to Philip's function]
    """ 
        The correction code is from here: https://github.com/BrancoLab/Common-Coordinate-Behaviour
    """
    matrix, points = get_matrix(videopath, maze_model=maze_model)

    # Return the updated key
    key['maze_model'] = maze_model
    key['correction_matrix'] = matrix
    key['alignment_points'] = points
    return key


def make_templates_table(key):
    """[allows user to define ROIs on the standard maze model that match the ROIs]
        """

    # Get all possible components name
    nplatf, nbridges = 6, 15
    platforms = ['p'+str(i) for i in range(1, nplatf + 1)]
    bridges = ['b'+str(i) for i in range(1, nbridges + 1)]
    components = ['s', 't']
    components.extend(platforms)
    components.extend(bridges)

    # Get maze model
    mmc = [m for m in CommonCoordinateMatrices if m['uid'] == key['uid']]
    if not mmc:
        raise ValueError(
            'Could not find CommonCoordinateBehaviour Matrix for this entry: ', key)
    else:
        model = mmc[0]['maze_model']

    # Load yaml with rois coordinates
    paths = load_yaml('paths.yml')
    rois = load_yaml(paths['maze_model_templates'])

    # return new key
    return {**key, **rois}


def make_recording_table(key):
    

    # See which software to use
    if key['uid'] < 184:
        software = 'behaviour'
    else:
        software = 'mantis'








