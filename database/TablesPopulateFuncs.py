import sys
sys.path.append('./')

import time
import datajoint as dj
from database.dj_config import start_connection
from nptdms import TdmsFile
import pandas as pd
import os
from collections import namedtuple
import numpy as np
import cv2
import warnings
import matplotlib.pyplot as plt
import scipy.signal as signal

from database.NewTablesDefinitions import *

from Utilities.file_io.files_load_save import load_yaml, load_tdms_from_winstore
from Utilities.video_and_plotting.commoncoordinatebehaviour import run as get_matrix
from Utilities.video_and_plotting.video_editing import Editor

from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame
from Processing.tracking_stats.extract_velocities_from_tracking import complete_bp_with_velocity, get_body_segment_stats
from Processing.tracking_stats.math_utils import *


""" 
    Collection of functions used to populate the dj.Import and dj.Compute
    tables defined in NewTablesDefinitions.py
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
        self.analog_input_folder = os.path.join(self.paths['raw_data_folder'], 
                                                self.paths['raw_analoginput_folder'])
        self.pose_folder = self.paths['tracked_data_folder']

    def get_behaviour_recording_files(self, session):
        raw_video_folder = self.raw_video_folder
        raw_metadata_folder = self.raw_metadata_folder

        # get video and metadata files
        videos = sorted([f for f in os.listdir(raw_video_folder)
                            if session['session_name'].lower() in f.lower() and 'test' not in f
                            and '.h5' not in f and '.pickle' not in f])
        metadatas = sorted([f for f in os.listdir(raw_metadata_folder)
                            if session['session_name'].lower() in f.lower() and 'test' not in f and '.tdms' in f])

        if videos is None or metadatas is None:
            raise FileNotFoundError(videos, metadatas)

        # Make sure we got the correct number of files, otherwise ask for user input
        if not videos or not metadatas:
            if not videos and not metadatas:
                print('Couldnt find filessss')
                return None, None
                # raise ValueError('Found no files for ', session['session_name'])
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


    def open_temp_tdms_as_df(self, path, move=True, skip_df=False):
        """open_temp_tdms_as_df [gets a file from winstore, opens it and returns the dataframe]
        
        Arguments:
            path {[str]} -- [path to a .tdms]
        """
        # Download .tdms from winstore, and open as a DataFrame
        # ? download from winstore first and then open, faster?
        if move:
            temp_file = load_tdms_from_winstore(path)
        else:
            temp_file = path

        print('opening ', temp_file, ' with size {} GB'.format(
            round(os.path.getsize(temp_file)/1000000000, 2)))
        bfile = open(temp_file, 'rb')
        tdmsfile = TdmsFile(bfile, memmap_dir="M:\\")
        print('     ... opened')
        if skip_df:
            return tdmsfile, None
        else:
            tdms_df = tdmsfile.as_dataframe()
            print('         ... as dataframe')
            # Extract data and insert in key
            cols = list(tdms_df.columns)
            return tdms_df, cols

    def extract_behaviour_stimuli(self, aifile):
        """extract_behaviour_stimuli [given the path to a .tdms file with session metadata extract
        stim names and timestamp (in frames)]
        
        Arguments:
            aifile {[str]} -- [path to .tdms file] 
        """
        # Get .tdms as a dataframe
        tdms_df, cols = self.open_temp_tdms_as_df(aifile, move=False)
        # ? Print out content of the dataframe
        # with open('behav_cols.txt', 'w+') as out:
        #     for c in cols:
        #         out.write(c+'\n\n')

        # Loop over the dataframe columns named like : 
        # /'Visual Stimulis'/' 20130-FC_slowloom'
        stim_cols = [c for c in cols if 'Stimulis' in c]
        stimuli = []
        stim = namedtuple('stim', 'type name frame')
        for c in stim_cols:
            stim_type = c.split(' Stimulis')[0][2:].lower()
            if 'digit' in stim_type: continue
            stim_name = c.split('-')[-1][:-2].lower()
            try:
                stim_frame = int(c.split("'/' ")[-1].split('-')[0])
            except:
                stim_frame = int(c.split("'/'")[-1].split('-')[0])
            stimuli.append(stim(stim_type, stim_name, stim_frame))
        return stimuli

    def extract_ai_info(self, key, aifile):
        """
        aifile: str path to ai.tdms

        extract channels values from file and returns a key dict for dj table insertion

        """

        # Get .tdms as a dataframe
        tdms_df, cols = self.open_temp_tdms_as_df(aifile, move=True, skip_df=True)
        chs = ["/'OverviewCameraTrigger_AI'/'0'", "/'ThreatCameraTrigger_AI'/'0'", "/'AudioIRLED_AI'/'0'", "/'AudioFromSpeaker_AI'/'0'"]
        """ 
        Now extracting the data directly from the .tdms without conversion to df
        """
        key['overview_camera_triggers'] = np.round(tdms_df.object('OverviewCameraTrigger_AI', '0').data, 2)
        key['threat_camera_triggers'] = np.round(tdms_df.object('ThreatCameraTrigger_AI', '0').data, 2)
        key['audio_irled'] = np.round(tdms_df.object('AudioIRLED_AI', '0').data, 2)
        if 'AudioFromSpeaker_AI' in tdms_df.groups():
            key['audio_signal'] = np.round(tdms_df.object('AudioFromSpeaker_AI', '0').data, 2)
        else:
            key['audio_signal'] = -1
        key['ldr'] = -1  # ? insert here
        key['tstart'] = -1
        key['manuals_names'] = -1
        # warnings.warn('List of strings not currently supported, cant insert manuals names')
        key['manuals_timestamps'] = -1 #  np.array(times)
        return key

""" 
##################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
##################################################
"""

def make_dlcmodels_table(table):
    """make_dlcmodels_table [Fills in dlc models table from dlcmodels.yml. making sure that
    only one model per camera is present in the table]
    
    Arguments:
        table {[class]} -- [dj table]
    """

    names_in_table = table.fetch('model_name')
    cameras_in_table = table.fetch('camera')
    models = load_yaml('dlcmodels.yml')
    
    for model in models.values():
        if model['camera'] in cameras_in_table:
            continue
            # one with the same camera is already present
            # if same name: overwrite
            # else: replace?
            if model['model_name'] in names_in_table:
                var = 'model_name'
                print('A model for camera {} with name {} exists already'.format(model['camera'], model['model_name']))
            else:
                var = 'camera'
                print('A model for camera {} already exists, replace?'.format(model['camera']))
            
            print('Old model: ', (table & '{}={}'.format(var, model[var])))
            print('New model: ', model)
            yn = input('Overwrite? [y/n]')
            if y != 'y': continue
            else:
                (table & '{}={}'.format(var, model[var]).delete())
                table.insert1(model)
        else:
            table.insert1(model)
    print(table)
    

def make_commoncoordinatematrices_table(table, key, sessions, videofiles, fast_mode=False):
    """make_commoncoordinatematrices_table [Allows user to align frame to model
    and stores transform matrix for future use]
    
    Arguments:
        key {[dict]} -- [key attribute of table autopopulate method from dj]
    """
    if fast_mode: # ? just do one session per day
        # If an entry with the same date exists already, avoid re doing the points mapping
        this_date = [s for s in sessions.fetch(as_dict=True) if s['uid']==key['uid']][0]['date']
        old_entries = [e for e in sessions.fetch(as_dict=True) if e['uid'] in table.fetch('uid')]
        
        if old_entries:
            old_entry = [o for o in old_entries if o['date']==this_date]
            if old_entry:
                old_matrix = [m for m in table.fetch(as_dict=True) if m['uid']==old_entry[0]['uid']][0]
                key['maze_model'] = old_matrix['maze_model']
                key['correction_matrix'] = old_matrix['correction_matrix']
                key['alignment_points'] = old_matrix['alignment_points']
                key['top_pad'] = old_matrix['top_pad']
                key['side_pad'] = old_matrix['top_pad']
                table.insert1(key)
                return

    # Get the maze model template
    maze_model = cv2.imread('Utilities\\video_and_plotting\\mazemodel.png')
    maze_model = cv2.resize(maze_model, (1000, 1000))
    maze_model = cv2.cv2.cvtColor(maze_model, cv2.COLOR_RGB2GRAY)

    # Get path to video of first recording
    rec = [r for r in videofiles if r['session_name']
            == key['session_name'] and r['camera_name']=='overview']

    if not rec:
        print('Did not find recording or videofiles while populating CCM table. Populate recordings and videofiles first! Session: ', key['session_name'])
        a = 1
        return
        # raise ValueError(
        #     'Did not find recording while populating CCM table. Populate recordings first! Session: ', key['session_name'])
    else:
        rec = rec[0]
        if not '.' in rec['converted_filepath']:
            videopath = rec['video_filepath']
        else:
            videopath = rec['converted_filepath']

    if 'joined' in videopath:
        raise ValueError


    # Apply the transorm [Call function that prepares data to feed to Philip's function]
    """ 
        The correction code is from here: https://github.com/BrancoLab/Common-Coordinate-Behaviour
    """
    matrix, points, top_pad, side_pad = get_matrix(videopath, maze_model=maze_model)
    if matrix is None:   # somenthing went wrong and we didn't get the matrix
        # Maybe the videofile wasn't there
        print('Did not extract matrix for video: ', videopath)
        return


    # Return the updated key
    key['maze_model'] = maze_model
    key['correction_matrix'] = matrix
    key['alignment_points'] = points
    key['top_pad'] = top_pad
    key['side_pad'] = side_pad
    table.insert1(key)


def make_templates_table(key, sessions, ccm):
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
    mmc = [m for m in ccm if m['uid'] == key['uid']]
    if not mmc:
        print('Could not find CommonCoordinateBehaviour Matrix for this entry: ', key)
        return
        # raise ValueError(
        #     'Could not find CommonCoordinateBehaviour Matrix for this entry: ', key)
    else:
        model = mmc[0]['maze_model']

    # Load yaml with rois coordinates
    paths = load_yaml('paths.yml')
    rois = load_yaml(paths['maze_model_templates'])

    # Only keep the rois relevant for each experiment
    sessions = pd.DataFrame(sessions().fetch())
    experiment = sessions.loc[sessions['uid'] == key['uid']].experiment_name.values[0]
    rois_per_exp = load_yaml(paths['maze_templates_per_experiment'])
    rois_per_exp = rois_per_exp[experiment]
    selected_rois = {k:(p if k in rois_per_exp else -1)  for k,p in rois.items()}

    # return new key
    return {**key, **selected_rois}


def make_recording_table(table, key):
    def behaviour(table, key, software):
        tb = ToolBox()
        videos, metadatas = tb.get_behaviour_recording_files(key)
        if videos is None: return
        # Loop over the files for each recording and extract info
        for rec_num, (vid, met) in enumerate(zip(videos, metadatas)):
            if vid.split('.')[0].lower() != met.split('.')[0].lower():
                raise ValueError('Files dont match!', vid, met)

            name = vid.split('.')[0]
            try:
                recnum = int(name.split('_')[2])
            except:
                recnum = 1

            if rec_num+1 != recnum:
                raise ValueError(
                    'Something went wrong while getting recording number within the session')

            rec_name = key['session_name']+'_'+str(recnum)
            
            # Insert into table
            rec_key = key.copy()
            rec_key['recording_uid'] = rec_name
            rec_key['ai_file_path'] = os.path.join(tb.raw_metadata_folder, met)
            rec_key['software'] = software
            table.insert1(rec_key)

    def mantis(table, key, software):
        # Get AI file and insert in Recordings table
        tb = ToolBox()
        rec_name = key['session_name']
        aifile = [os.path.join(tb.analog_input_folder, f) for f 
                    in os.listdir(tb.analog_input_folder) 
                    if rec_name in f]
        if not aifile:
            print('aifile not found for session: ', key, '\n\n')
            return
        else:
            aifile = aifile[0]
        
        key_copy = key.copy()
        key_copy['recording_uid'] = rec_name
        key_copy['software'] = software
        key_copy['ai_file_path'] = aifile
        table.insert1(key_copy)

        # Extract info from aifile and populate part table
        ai_key = tb.extract_ai_info(key, aifile)
        ai_key['recording_uid'] = rec_name

        print('Key of size: ', sys.getsizeof(ai_key))
        table.AnalogInputs.insert1(ai_key)
        
        print('Succesfully inserted into mantis table')

    # See which software was used and call corresponding function
    print(' Processing: ', key)
    if key['uid'] < 184:
        software = 'behaviour'
        behaviour(table, key, software)
    else:
        software = 'mantis'
        mantis(table, key, software)


def make_videofiles_table(table, key, recordings, videosincomplete):
    def make_videometadata_table(filepath, key):
        # Get videometadata
        cap = cv2.VideoCapture(filepath)
        key['tot_frames'], fps, key['frame_height'], key['fps'] = Editor.get_video_params(
            cap)
        key['frame_width'] = np.int(fps)
        key['frame_size'] =  key['frame_width']* key['frame_height']
        key['camera_offset_x'], key['camera_offset_y'] = -1, -1

        # if key['fps'] < 10: raise ValueError('Couldnt get metadata for ', filepath, key)

        return key

    def behaviour(table, key, videosincomplete):
        tb  = ToolBox()
        videos, metadatas = tb.get_behaviour_recording_files(key)
        
        if key['recording_uid'].count('_') == 1:
            recnum = 1
        else:
            rec_num = int(key['recording_uid'].split('_')[-1])
        rec_name = key['recording_uid']
        try:
            vid, met = videos[rec_num-1], metadatas[rec_num-1]
            vid, met = os.path.join(tb.raw_video_folder, vid), os.path.join(tb.raw_metadata_folder, vid)
        except:
            raise ValueError('Could not collect video and metadata files:' , rec_num-1, rec_name, videos)
        # Get deeplabcut data
        posefile = [os.path.join(tb.tracked_data_folder, f) for f in os.listdir(tb.tracked_data_folder)
                    if rec_name == os.path.splitext(f)[0].split('_pose')[0] and '.pickle' not in f]
        
        if not posefile:
            new_rec_name = rec_name[:-2]
            posefile = [os.path.join(tb.tracked_data_folder, f) for f in os.listdir(tb.tracked_data_folder)
                        if new_rec_name == os.path.splitext(f)[0].split('_pose')[0] and '.pickle' not in f]

        if not posefile:
            # ! pose file was not found, create entry in incompletevideos table to mark we need dlc analysis on this
            incomplete_key = key.copy()
            incomplete_key['camera_name'] = 'overview'
            incomplete_key['conversion_needed'] = 'false'
            incomplete_key['dlc_needed'] = 'true'
            # try:
            #     videosincomplete.insert1(incomplete_key)
            # except:
            #     print(videosincomplete.describe())
            #     raise ValueError('Could not insert: ', incomplete_key )

            # ? Create dummy posefile name which will be replaced with real one in the future
            vid_name, ext = vid.split('.')
            posefile = vid_name+'_pose'+ '.h5'

        elif len(posefile) > 1:
            raise FileNotFoundError('Found too many pose files: ', posefile)
        else:
            posefile = posefile[0]

        # Insert into Recordings.VideoFiles 
        new_key = key.copy()
        new_key['camera_name'] = 'overview'
        new_key['video_filepath'] = vid
        new_key['converted_filepath'] = 'nan'
        new_key['metadata_filepath'] = 'nan'
        new_key['pose_filepath'] = posefile
        table.insert1(new_key)

        return vid

    def mantis(table, key, videosincomplete):
        def insert_for_mantis(table, key, camera, vid, conv, met, pose):
            to_remove = ['tot_frames', 'frame_height', 'frame_width', 'frame_size',
                        'camera_offset_x', 'camera_offset_y', 'fps']
            
            video_key = key.copy()
            video_key['camera_name'] = camera
            video_key['video_filepath'] = vid
            video_key['converted_filepath'] = conv
            video_key['metadata_filepath'] = met
            video_key['pose_filepath'] = pose
            if 'conversion_needed' in video_key.keys():
                del video_key['conversion_needed'], video_key['dlc_needed']
            try:
                kk = tuple(video_key.keys())
                for n in to_remove:
                    if n in kk: del video_key[n]
                table.insert1(video_key)
            except:
                raise ValueError('Could not isnert ', video_key)
            
            metadata_key = make_videometadata_table(video_key['converted_filepath'], key)
            if 'conversion_needed' in metadata_key.keys():
                del metadata_key['conversion_needed'], metadata_key['dlc_needed']
            try:
                table.Metadata.insert1(metadata_key)
            except:
                return
            
        def check_files_correct(ls, name):
            """check_files_correct [check if we found the expected number of files]
            
            Arguments:
                ls {[list]} -- [list of file names]
                name {[str]} -- [name of the type of file we are looking for ]
            
            Raises:
                FileNotFoundError -- [description]
            
            Returns:
                [bool] -- [return true if everuything is fine else is false]
            """

            if not ls:
                print('Did not find ', name)
                return False
            elif len(ls)>1:
                raise FileNotFoundError('Found too many ', name, ls)
            else:
                return True

        def add_videosincomplete_entry(videosincomplete, ikey, vid, converted_check, pose_check):
            """add_videosincomplete_entry [adds entry to videos incompelte table to mark that stuff needs to be done ]
            
            Arguments:
                videosincomplete {[obj]} -- [dj table]
                key {[dict]} -- [key]
                vid {[str]} -- [name of video]
                converted_check {[bool]} -- [conversion needed]
                pose_check {[bool]} -- [dlc eneeded]
            """

            tokeep = ['recording_uid', 'uid', 'session_name']
            kk = tuple(ikey.keys())
            for k in kk:
                if k not in tokeep: del ikey[k]

            cameras = ['overview', 'threat', 'catwalk', 'top_mirror', 'side_mirror']
            camera = [c for c in cameras if c in vid.lower()]
            if not camera:
                raise ValueError('sometihngs wrong ', vid)
            else:
                camera = camera[0]
            key['camera_name'] = camera
            if converted_check:
                ikey['conversion_needed'] = 'false'
            else:
                ikey['conversion_needed'] = 'true'
            if pose_check:
                ikey['dlc_needed'] = 'false'
            else:
                ikey['dlc_needed'] = 'true'
            # try:
            #     videosincomplete.insert1(ikey)
            # except:
            #     raise ValueError('Could not insert ', ikey, 'in', videosincomplete.heading)

        #############################################################################################

        tb = ToolBox()  # toolbox

        # Get video Files
        videos = [f for f in os.listdir(tb.raw_video_folder)
                        if 'tdms' in f and key['session_name'] in f]
        
        # Loop and get matching files
        for vid in videos:
            # Get videos
            videoname, ext = vid.split('.')
            converted = [f for f in os.listdir(tb.raw_video_folder)
                        if videoname in f and '.mp4' in f]
            converted_check = check_files_correct(converted, 'converted')
            if converted_check: converted = converted[0]

            metadata = [f for f in os.listdir(tb.raw_metadata_folder)
                        if videoname in f and 'tdms' in f]
            metadata_check = check_files_correct(metadata, 'metadata')
            if not metadata_check: raise FileNotFoundError('Could not find metadata file!!')
            else: metadata = metadata[0]

            posedata = [os.path.splitext(f)[0].split('_pose')[0]+'.h5' 
                        for f in os.listdir(tb.pose_folder)
                        if videoname in f and 'h5' in f]
            pose_check = check_files_correct(posedata, 'pose data')
            if pose_check: posedata = posedata[0]
            
            # Check if anything is missing            
            if not converted_check or not pose_check:
                # add_videosincomplete_entry(videosincomplete, key, vid, converted_check, pose_check)
                # ? add dummy files names which will be replaced with real ones in the future
                if not converted_check:
                    converted = videoname+'.mp4'
                if not pose_check:
                    posedata = videoname+'_pose.h5'

            # Get Camera Name and views videos
            if 'Overview' in vid:
                camera = 'overview'
            elif 'Threat' in vid:
                camera = 'threat'

                # ? work on views videos
                # Get file names and create cropped videos if dont exist
                ed = Editor()
                catwalk, side, top = ed.mirros_cropper(os.path.join(tb.raw_video_folder,vid),
                                                            os.path.join(tb.raw_video_folder, 'Mirrors'))
                views_videos = [catwalk, side, top]
                views_names = ['catwalk', 'side_mirror', 'top_mirror']

                # Get pose names
                views_poses = {}
                for vh, view_name in zip(views_videos, views_names):
                    n = os.path.split(vh)[-1].split('.')[0]
                    pd = [os.path.splitext(f)[0].split('_pose')[0]+'.h5'
                                for f in os.listdir(os.path.join(tb.pose_folder, 'Mirrors'))
                                if n in f and 'h5' in f]
                    
                    pd_check = check_files_correct(pd, 'cropped video pose file')
                    if not pd_check:  
                        add_videosincomplete_entry(videosincomplete, key, view_name, True, pd_check)
                        # ? add dummy file name 
                        pd = n+'_pose.h5'
                    else: pd = pd[0]
                    views_poses[view_name] = pd

                # Insert into table [video and converted are the same here]
                view = namedtuple('view', 'camera video metadata pose')
                views = [view('catwalk', catwalk, 'nan', views_poses['catwalk']), 
                        view('side_mirror', side, 'nan', views_poses['side_mirror']), 
                        view('top_mirror', top, 'nan', views_poses['top_mirror'])]
                
                for insert in views:
                    insert_for_mantis(table, key, insert.camera, insert.video,
                                        insert.video, insert.metadata, insert.pose)
            else:
                raise ValueError('Unexpected videoname ', vid)

            # Insert Main Video (threat or overview) in table
            insert_for_mantis(table, key, camera, os.path.join(tb.raw_video_folder, vid),
                                os.path.join(tb.raw_video_folder, converted),
                                os.path.join(tb.raw_metadata_folder, metadata),
                                os.path.join(tb.pose_folder, posedata))

    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################

    print('Processing:  ', key)
    # Call functions to handle insert in main table
    software = [r['software'] for r in recordings.fetch(as_dict=True) if r['recording_uid']==key['recording_uid']][0]
    if not software:
        raise ValueError()
    if software == 'behaviour':
        videopath = behaviour(table, key, videosincomplete)
        # Insert into part table
        metadata_key = make_videometadata_table(videopath, key)
        metadata_key['camera_name'] = 'overview'
        table.Metadata.insert1(metadata_key)
    else:
        videopath = mantis(table, key, videosincomplete)

    
def make_behaviourstimuli_table(table, key, recordings, videofiles):
    if key['uid'] > 184:
        print(key['recording_uid'], '  was not recorded with behaviour software')
        return
    else:
        print('Extracting stimuli info for recording: ', key['recording_uid'])

    # Get file paths    
    rec = [r for r in recordings.fetch(as_dict=True) if r['recording_uid']==key['recording_uid']][0]
    tdms_path = rec['ai_file_path']
    vid = [v for v in videofiles.fetch(as_dict=True) if v['recording_uid']==key['recording_uid']][0]
    videopath = vid['video_filepath']

    # Get stimuli
    tb = ToolBox()
    stimuli = tb.extract_behaviour_stimuli(tdms_path)

    # If no sti add empty entry to table to avoid re-loading everyt time pop method called
    if not stimuli:
        print('Adding fake placeholder entry')
        stim_key = key.copy()
        stim_key['stimulus_uid'] = key['recording_uid']+'_{}'.format(0)
        stim_key['stim_duration']  = -1
        stim_key['video'] = videopath
        stim_key['stim_type'] = 'nan'
        stim_key['stim_start'] = -1
        stim_key['stim_name'] = -1
        table.insert1(stim_key)
    else:
        # Add in table
        for i, stim in enumerate(stimuli):
            stim_key = key.copy()
            stim_key['stimulus_uid'] = key['recording_uid']+'_{}'.format(i)

            if 'audio' in stim.name: stim_key['stim_duration'] = 9 # ! hardcoded
            else: stim_key['stim_duration']  = 5
            
            stim_key['video'] = videopath
            stim_key['stim_type'] = stim.type
            stim_key['stim_start'] = stim.frame
            stim_key['stim_name'] = stim.name
            table.insert1(stim_key)



def make_mantistimuli_table(table, key, recordings, videofiles):
    def plot_signals(audio_channel_data, stim_start_times, overview=False, threat=False):
        f, ax = plt.subplots()
        ax.plot(audio_channel_data)
        ax.plot(stim_start_times, audio_channel_data[stim_start_times], 'x', linewidth=.4, label='audio')
        if overview:
            ax.plot(audio_channel_data, label='overview')
        if threat:
            ax.plot(audio_channel_data, label='threat')
        ax.legend()
        ax.set(xlim=[stim_start_times[0]-5000, stim_start_times[0]+5000])

    if key['uid'] <= 184:
        return
    else:
            print('Populating mantis stimuli for: ', key['recording_uid'])


    tb = ToolBox()
    rec = [r for r in recordings if r['recording_uid']==key['recording_uid']][0]
    aifile = rec['ai_file_path']

    # Get stimuli names from the ai file
    tdms_df, cols = tb.open_temp_tdms_as_df(aifile, move=True, skip_df=True)

    # Get stimuli
    groups = tdms_df.groups()

    if 'WAVplayer' in groups:
        stimuli_groups = tdms_df.group_channels('WAVplayer')
    else:
        stimuli_groups = tdms_df.group_channels('AudioIRLED_analog')
    stimuli = {s.path:s.data[0] for s in stimuli_groups}

    # Check if there is any stim
    if not len(stimuli.keys()):
        # There were no stimuli, let's insert a fake one to avoid loading the same files over and over again
        print(' No stim detected, inserting fake place holder')
        stim_key = key.copy()
        stim_key['stimulus_uid'] = stim_key['recording_uid']+'_{}'.format(0)
        stim_key['overview_frame'] = -1
        stim_key['threat_frame'] = -1
        stim_key['duration'] = -1 
        stim_key['overview_frame_off'] =  -1
        stim_key['threat_frame_off'] = -1
        stim_key['stim_name'] = 'nan'
        stim_key['stim_type'] = 'nan' 

        table.insert1(stim_key)
        return

    # Get stim times from audio channel data
    if  'AudioFromSpeaker_AI' in groups:
        audio_channel_data = tdms_df.channel_data('AudioFromSpeaker_AI', '0')
        # stim_start_times, _ = signal.find_peaks(audio_channel_data, height=.2, distance=9.1*sampling_rate, width=(1, 100), wlen=100)  # ! hardcoded miimal distance: duration * sampling rate
        th = 1
    else:
        # First recordings with mantis had different params
        audio_channel_data = tdms_df.channel_data('AudioIRLED_AI', '0')
        th = 1.5
    
    sampling_rate = 25000
    above_th = np.where(audio_channel_data>th)[0]
    peak_starts = [x+1 for x in np.where(np.diff(above_th)>sampling_rate)]
    stim_start_times = above_th[peak_starts]
    try:
        stim_start_times = np.insert(stim_start_times, 0, above_th[0])
    except:
        raise ValueError
        return

    # ? to visualise the finding of stim times over the audio channel:
    # ThreatCameraTrigger_AI = tdms_df.channel_data('ThreatCameraTrigger_AI', '0')
    # OverviewCameraTrigger_AI = tdms_df.channel_data('OverviewCameraTrigger_AI', '0')
    # plot_signals(audio_channel_data, stim_start_times, overview=OverviewCameraTrigger_AI, threat=ThreatCameraTrigger_AI)
    # plt.show()

    # Chck we found the correct number of peaks
    if not len(stimuli) == len(stim_start_times):
        print('Names - times: ', len(stimuli), len(stim_start_times),stimuli.keys(), stim_start_times)
        sel = input('Which to discard? ["n" if youd rather look at the plot]')
        if not 'n' in sel:
            sel = int(sel)
        else:
            plot_signals(audio_channel_data, stim_start_times)
            plt.show()
            sel = input('Which to discard? ')
        if len(stim_start_times) > len(stimuli):
            np.delete(stim_start_times, int(sel))
        else:
            del stimuli[list(stimuli.keys())[sel]]

    if not len(stimuli) == len(stim_start_times):
        raise ValueError

    # Go from stim time in number of samples to number of frames
    fps_overview = 40
    fps_threat = 120
    overview_stimuli_frames = np.round(np.multiply(np.divide(stim_start_times, sampling_rate), fps_overview))
    threat_stimuli_frames = np.round(np.multiply(np.divide(stim_start_times, sampling_rate), fps_threat))

    for i, (stimname, stim_protocol) in enumerate(stimuli.items()):
        stim_key = key.copy()
        stim_key['stimulus_uid'] = stim_key['recording_uid']+'_{}'.format(i)
        stim_key['overview_frame'] = int(overview_stimuli_frames[i])
        stim_key['threat_frame'] = int(threat_stimuli_frames[i])
        stim_key['duration'] = 9 # ! hardcoded
        stim_key['overview_frame_off'] =  int(overview_stimuli_frames[i]) + fps_overview*stim_key['duration'] # ! hardcoded!
        stim_key['threat_frame_off'] = int(threat_stimuli_frames[i]) + fps_threat*stim_key['duration'] # ! hardcoded!
        stim_key['stim_name'] = stimname
        stim_key['stim_type'] = 'audio' # ! hardcoded
        
        try:
            table.insert1(stim_key)
        except:
            raise ValueError('Cold not insert {} into {}'.format(stim_key, table.heading))


#####################################################################################################################
#####################################################################################################################


def make_trackingdata_table(table, key, videofiles, ccm_table, templates, sessions, fast_mode=False):
    if key['camera_name'] != 'overview': return

    # Get the name of the experiment the video belongs to
    fetched_sessions = sessions.fetch(as_dict=True)
    session = [s for s in fetched_sessions if s['uid']==key['uid']][0]
    experiment = session['experiment_name']

    if 'lambda' in experiment.lower(): return  # ? skip this useless experiment :)
    # if 'Model Based' not in experiment: return  #!

    fast_mode = fast_mode # ! fast MODE
    to_include = dict(
            bodyparts=['snout', 'neck', 'body', 'tail_base' , 'left_ear', 'right_ear'],
            segments = []
            # segments=['head', 'body_upper', 'body_lower']
    )


    # Check if we have all the data necessary to continue 
    try:
        vid, ccm = None, None
        vid = [v for v in videofiles.fetch(as_dict=True) if v['recording_uid'] == key['recording_uid']][0]
        ccm = [c for c in ccm_table.fetch(as_dict=True) if c['uid']==key['uid']][0]
    except:
        if vid is None:  print('Could not find videofile for ', key['recording_uid']) 
        else:  print('Could not find common coordinate matrix for ', key['recording_uid']) 
        return
    else:
        print('Processing tracking data for : ', key['recording_uid'])
    
    
    
    # Load the .h5 file with the tracking data 
    try:
        if not 'pose' in os.path.split(vid['pose_filepath'])[-1]:
            vid['pose_filepath'] = vid['pose_filepath'].split(".")[0]+"_pose.h5"

        posedata = pd.read_hdf(vid['pose_filepath'])
    except:
        print('Could not load pose data:', vid['pose_filepath'])
        return

    # Insert entry into MAIN CLASS for this videofile
    table.insert1(key)

    # Get the scorer name and the name of the bodyparts
    first_frame = posedata.iloc[0]
    bodyparts = first_frame.index.levels[1]
    scorer = first_frame.index.levels[0]

    """
        Loop over bodyparts and populate Bodypart Part table
    """
    bp_data = {}
    for bp in bodyparts:
        if fast_mode and  bp != 'body': continue
        elif not fast_mode and bp not in to_include['bodyparts']: continue
        print('     ... body part: ', bp)

        # Get XY pose and correct with CCM matrix
        xy = posedata[scorer[0], bp].values[:, :2]
        corrected_data = correct_tracking_data(xy, ccm['correction_matrix'], ccm['top_pad'], ccm['side_pad'], experiment, session['uid'])
        temp_dict = {}
        temp_dict['x'] = corrected_data[:, 0]
        temp_dict['y'] = corrected_data[:, 1]
        corrected_data = pd.DataFrame.from_dict(temp_dict)

        # get velocity
        vel = calc_distance_between_points_in_a_vector_2d(corrected_data.values)

        # get orientation [angle between XY at t0 ad XY at t1]
        theta = calc_angle_between_points_of_vector(corrected_data.values)

        # get distance from shelter
        shelter = (500, 740)
        shelter_dist = calc_distance_from_shelter(corrected_data.values, shelter)

        # Add new vals
        corrected_data['velocity'] = vel
        corrected_data['angle'] = theta
        corrected_data['shelter_distance'] = shelter_dist

        # If bp is body get the position on the maze
        if 'body' in bp:
            # Get position of maze templates - and shelter
            templates_idx = [i for i, t in enumerate(templates.fetch()) if t['uid'] == key['uid']][0]
            rois = pd.DataFrame(templates.fetch()).iloc[templates_idx]

            del rois['uid'], rois['session_name'], 

            # Calcualate in which ROI the body is at each frame - and distance from the shelter
            corrected_data['roi_at_each_frame'] = get_roi_at_each_frame(experiment, key['recording_uid'], corrected_data, dict(rois))  # ? roi name
            # ! dj limitation here
            warnings.warn('Currently DJ canot store string of lists so roi_at_each_Frame is not saved in the databse')
            rois_ids = {p:i for i,p in enumerate(rois.keys())}  # assign a numeric value to each ROI
            corrected_data['roi_at_each_frame'] = np.array([rois_ids[r] for r in corrected_data['roi_at_each_frame']])
            
        # Insert into part table
        bp_data[bp] = corrected_data
        bpkey = key.copy()
        bpkey['bpname'] = bp
        bpkey['tracking_data'] = corrected_data.values 
        try:
            table.BodyPartData.insert1(bpkey)
        except:
            pass

    """
        Loop over body segments and populate body semgents Part table
    """
    # if not fast_mode:
    #     body_axis = []
    #     for segment_name, (bp1, bp2) in table.segments.items():
    #         if segment_name not in to_include['segments']: continue
    #         print('     ... body segment: ', segment_name)
    #         # get position of each bodypart as numpy array
    #         bp1_data = np.array([bp_data[bp1]['x'], bp_data[bp1]['y']])
    #         bp2_data = np.array([bp_data[bp2]['x'], bp_data[bp2]['y']])

    #         # Create dataframe with segment data and convert to dataframe
    #         segment_data = {}
    #         segment_data['length'] = calc_distance_between_points_two_vectors_2d(bp1_data.T, bp2_data.T)
    #         try:
    #             segment_data['theta'] = calc_angle_between_vectors_of_points_2d(bp1_data, bp2_data)  
    #             segment_data['angvel'] = calc_ang_velocity(segment_data['theta'])
    #         except:
    #             warnings.warn('Could not extract theta')
    #             segment_data['theta'] = np.zeros((len(segment_data['length'])))
    #             segment_data['angvel'] = np.zeros((len(segment_data['length'])))

    #         if segment_name in ['head', 'body_upper', 'body_lower']:
    #             body_axis.append(np.array(segment_data['length']))

    #         segment_data_df = pd.DataFrame.from_dict(segment_data)
    #         # Insert into part table
    #         segment_key = key.copy()
    #         segment_key['bp1'] = bp1
    #         segment_key['bp2'] = bp2
    #         segment_key['tracking_data'] = segment_data_df.values 

    #         table.BodySegmentData.insert1(segment_key)


    #     # Calculate body length and insert it into table
    #     body_axis_length = np.add(body_axis[0], body_axis[1])
    #     body_axis_length = np.add(body_axis_length, body_axis[2])

    #     temp_key = key.copy()
    #     temp_key['bp1'] = 'body_axis'
    #     temp_key['bp2'] = 'body_axis'
    #     temp_key['tracking_data'] = body_axis_length






#####################################################################################################################
#####################################################################################################################

def populate_ArmsProbs():
    from database.NewTablesDefinitions import ArmsProbs, Sessions, AllTrials
    to_fetch = ["origin_arm", "escape_arm", "experiment_name", "number_of_trials"]
    table = ArmsProbs()
    lookup = table.arms_lookup_f()

    # Loop over each session but skip the ones that are already in the table
    uids, session_names = Sessions.fetch("uid", "session_name")
    sessions_in_table = table.fetch("uid")

    for n, (uid, session_name) in enumerate(zip(uids, session_names)):
        print("Processing {} of {} - {}".format(n, len(uids)-1, session_name))
        if uid in sessions_in_table: 
            print("     ... already in table")
            continue

        # Fetch the data for the session's trials
        origins, escapes, experiment_names, number_of_trials = (AllTrials & "is_escape='true'" & "session_uid='{}'".format(uid)).fetch(*to_fetch)

        # check if there is any trial from this session
        if not np.any(experiment_names): continue

        """
            Insert into main table
        """
        # Convert escape and origin arms to integers array
        escape_arms = np.array([lookup[a] for a in escapes])
        origin_arms = np.array([lookup[a] for a in origins])

        key = dict(
            uid = uid, 
            session_name = session_name, 
            experiment_name = experiment_names[0],
            escape_arms = escape_arms,
            origin_arms =  origin_arms, 
            n_escapes = len(escapes),
            n_trials = number_of_trials[0]
        )

        table.insert1(key)

        """
            Insert into parts tables
        """

        # Get a set of all the arms
        all_arms = []
        all_arms.extend(origins)
        all_arms.extend(escapes)
        arms = set(all_arms)

        # For each arm, look at the probability associated and populate part table
        for arm in arms:
            escape_p = calc_prob_item_in_list(escapes, arm)
            origin_p = calc_prob_item_in_list(origins, arm)
            n_times_escapes = len([x for x in escapes if x == arm])

            part_key = dict(
                uid = uid,
                session_name = session_name,
                arm_name = arm,
                escape_p = escape_p,
                origin_p = origin_p,
                n_times_taken = n_times_escapes
            )

            # Insert into part table
            table.Arm.insert1(part_key)




