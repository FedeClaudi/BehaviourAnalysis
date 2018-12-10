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
from Utilities.video_and_plotting.video_editing import get_video_params


""" 
    Collection of functions used to populate the dj.Import and dj.Compute
    tables defined in NewTablesDefinitions.py

    CommonCoordinateMatrices    ok
    Templates                   ok
    Recordings                  ok
    VideoFiles                  ok
    VideoMetadata               ok
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
        self.analog_input_folder = os.path.join(self.paths['raw_dat_folder'], 
                                                self.paths['raw_analoginput_folder'])

        self.pose_folder = paths['tracked_data_folder']

    def get_behaviour_recording_files(self, session):
        raw_video_folder = self.raw_video_folder
        raw_metadata_folder = self.raw_metadata_folder

        # get video and metadata files
        videos = sorted([f for f in os.listdir(raw_video_folder)
                            if session['session_name'].lower() in f.lower() and 'test' not in f
                            and '.h5' not in f and '.pickle' not in f])
        metadatas = sorted([f for f in os.listdir(raw_metadata_folder)
                            if session['session_name'].lower() in f.lower() and 'test' not in f and '.tdms' in f])

        # Make sure we got the correct number of files, otherwise ask for user input
        if not videos or not metadatas:
            if not videos and not metadatas:
                return
            print('couldnt find files for session: ',
                    session['session_name'])
            raise FileNotFoundError('dang')
        else:
            if len(videos) != len(metadatas):
                print('Found {} videos files: {}'.format(len(videos), videos))
                print('Found {} metadatas files: {}'.format(
                    len(metadatas), metadatas))
                raise ValueError(
                    'Something went wront wrong trying to get the files')

            num_recs = len(videos)
            print(' ... found {} recs'.format(num_recs))
            return videos, metadatas


    def extract_ai_info(self, key, aifile):
        raise NotImplementedError
        return {}

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


def make_recording_table(table, key):
    def behaviour(table, key, software):
        tb = ToolBox()
        videos, metadatas = tb.get_behaviour_recording_files(key)
        
        # Loop over the files for each recording and extract info
        for rec_num, (vid, met) in enumerate(zip(videos, metadatas)):
            if vid.split('.')[0].lower() != met.split('.')[0].lower():
                raise ValueError('Files dont match!')

            name = vid.split('.')[0]
            try:
                recnum = int(name.split('_')[2])
            except:
                recnum = 1

            if rec_num+1 != recnum:
                raise ValueError(
                    'Something went wrong while getting recording number within the session')

            rec_name = session['session_name']+'_'+str(recnum)
            
            # Insert into table
            rec_key = key.copy()
            rec_key['recording_uid'] = rec_name
            rec_key['ai_file_path'] = 'nan'
            rec_key['software'] = software
            table.insert1(rec_key)


    def mantis(table, key, software):
        # Get AI file and insert in Recordings table
        tb = ToolBox()
        rec_name = key['session_name']
        aifile = [os.path.join(tb.analog_input_folder, f) for f 
                  in os.listdir(tb.analog_input_folder) 
                  if rec_name in f][0]
        
        key['recording_uid'] = rec_name
        key['software'] = software
        key['ai_file_path'] = aifile

        table.insert1(key)

        # Extract info from aifile and populate part table
        ai_key = tb.extract_ai_info(key, aifile)
        Recordings.AnalogInputs.insert1(ai_key)

    # See which software was used and call corresponding function
    if key['uid'] < 184:
        software = 'behaviour'
        behaviour(table, key, software)
    else:
        software = 'mantis'
        mantis(table, key, software)


def make_videofiles_table(table, key):
    def make_videometadata_table(filepath, key):
        # Get videometadata
        cap = cv2.VideoCapture(filepath)
        key['tot_frames'], key['width'], key['height'], key['fps'] = get_video_params(
            cap)
        key['frame_size'] = width*height
        key['camera_offset_x'], key['camera_offset_y'] = -1, -1
        return key

    def behaviour(table, key):
        tb  = ToolBox()
        videos, metadatas = tb.get_behaviour_recording_files()
        
        rec_num = int(key['recording_uid'].split('_')[-1])
        rec_name = key['recording_uid']
        vid, met = videos[rec_num], metadats[rec_num]

        # Get deeplabcut data
        posefile = [os.path.join(tb.tracked_data_folder, f) for f in os.listdir(tb.tracked_data_folder)
                    if rec_name == os.path.splitext(f)[0].split('Deep')[0] and '.pickle' not in f]
        
        if not posefile:
            new_rec_name = rec_name[:-2]
            posefile = [os.path.join(tb.tracked_data_folder, f) for f in os.listdir(tb.tracked_data_folder)
                        if new_rec_name == os.path.splitext(f)[0].split('Deep')[0] and '.pickle' not in f]

        if not posefile or len(posefile)>1:
            raise FileNotFoundError('Pose files found: ', posefile)
        else:
            posefile = posefile[0]

        # Insert into Recordings.VideoFiles 
        new_key = key.copy()
        new_key['camera_name'] = 'overview'
        new_key['video_filepath'] = vid
        new_key['converted_filepath'] = 'nan'
        new_key['metadata_filepath'] = 'nan'
        new_key['posefilepath'] = posefile
        table.insert1(new_key)

        return vid

    def mantis(table, key):
        def insert_for_mantis(table, key, camera, vid, conv, met, pose):
            video_key = key.copy()
            video_key['camera_name'] = camera
            video_key['video_filepath'] = vid
            video_key['converted_filepath'] = conv
            video_key['metadata_filepath'] = met
            video_key['pose_filepath'] = pose
            table.insert1(video_key)

            metadata_key = make_videometadata_table(video_key['converted_filepath'], key)
            VideoFiles.VideoMetadata.insert1(metadata_key)


        tb = ToolBox()

        # Get video Files
        videos = [f for f in os.listdir(tb.raw_video_folder)
                        if 'tdms' in f and key['session_name'] in f]
        
        # Loop and get matching files
        for vid in videos:
            videoname = vid.split('.')[0]

            converted = [f for f in os.listdir(tb.raw_video_folder)
                        if videoname in f and '__joined' in f]
            if not converted or len(converted)> 1:
                raise FileNotFoundError('Converted videos ', converted)

            metadata = [f for f in os.listdir(tb.raw_metadata_folder)
                         if videoname in f and 'tdms' in f]
            if not metadata or len(metadata) > 1:
                raise FileNotFoundError('Metadata videos ', converted)

            posedata = [os.path.splitext(f)[0].split('Deep')[0]+'.h5' 
                        for f in os.listdir(tb.pose_folder)
                        if videoname in f and 'h5' in f]
            if not posedata or len(posedata) > 1:
                raise FileNotFoundError('Metadata videos ', converted)
            
            # Get Camera Name
            if 'Overview' in vid:
                camera = 'overview'
            elif 'Threat' in vid:
                camera = 'threat'
                # Get info and insert views file in table
                raise NotImplementedError
            else:
                raise ValueError('Unexpected videoname ', vid)

            # Insert Main Video (threat or overview) in table
            insert_for_mantis(table, key, camera, os.path.join(tb.raw_video_folder, vid),
                              os.path.join(tb.raw_video_folder, converted[0]),
                              os.path.join(tb.raw_metadata_folder, metadata[0]),
                              os.path.join(tb.pose_folder, posedata[0]))


    # Call functions to handle insert in main table
    if key['software'] == 'behaviour':
        videopath = behaviour(table, key)
        # Insert into part table
        metadata_key = make_videometadata_table(videopath, key)
        VideoFiles.VideoMetadata.insert1(metadata_key)
    else:
        videopath = mantis(table, key)

    
    
                                                                                                                        rec_name, posefile))

    











