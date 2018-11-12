from dj_config import start_connection
start_connection()

import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project

import os
import warnings
from shutil import copyfile

import cv2
import datajoint as dj
import moviepy
import pandas as pd
import pyexcel
import yaml
from moviepy.editor import VideoFileClip
from tqdm import tqdm

from Tables_definitions import *
from Utilities.video_editing import *
from Utilities.stim_times_loader import *


class PopulateDatabase:
    def __init__(self):
        """
        Collection of methods to populate the database
        """
        print("""
        Ready to populate database. Available classes:
                * Mouse
                * Experiment
                * Surgery
                * Manipulation
                * Session
            Updating SESSION will also update:
                * NeuralRecording
                * BehaviourRecording
                * BehaviourTrial""")

        # Hard coded paths to relevant files and folders
        with open('database\data_paths.yml', 'r') as f:
            paths = yaml.load(f)

        self.mice_records = paths['mice_records']
        self.exp_records = paths['exp_records']

        self.raw_data_folder = paths['raw_data_folder']
        self.raw_to_sort = os.path.join(self.raw_data_folder, paths['raw_to_sort'])
        self.raw_metadata_folder = os.path.join(self.raw_data_folder, paths['raw_metadata_folder'])
        self.raw_video_folder = os.path.join(self.raw_data_folder, paths['raw_video_folder'])

        self.trials_clips = os.path.join(self.raw_data_folder, paths['trials_clips'])
        self.tracked_data_folder = paths['tracked_data_folder']

        self.mice = Mice()
        self.sessions = Sessions()
        self.recordings = Recordings()
        self.trials = Trials()
        self.all_tables = dict(mice=self.mice,sessions= self.sessions, recordings=self.recordings, trials=self.trials)

    def display_tables_headings(self):
        for name, table in self.all_tables.items():
            print('\n\nTable definition for {}:\n{}'.format(name, table))

    def reset_database(self):
        print('ATTENTION: this might result in loss of data!!!')
        q = input('Continue ? [Y/N]')
        if not q.lower() == 'y':
            return
        else:
            [table.drop() for table in self.all_tables.values()]

    def populate_mice_table(self):
        """ Populates the Mice() table from the database"""
        """
          mouse_id: varchar(128)                        # unique mouse id
          ---
          strain:   varchar(128)                        # genetic strain
          dob: varchar(128)                             # mouse date of birth 
          sex: enum('M', 'F', 'U')                      # sex of mouse - Male, Female, or Unknown/Unclassified
          single_housed: enum('Y', 'N')                 # single housed or group caged
          enriched_cage: enum('Y', 'N')                 # presence of wheel or other stuff in the cage
        """

        table = self.mice
        loaded_excel = pyexcel.get_records(file_name=self.mice_records)

        for m in loaded_excel:
            if not m['']: continue
            inputdata = (m[''], m['Strain'], m['DOB'].strip(), 'M', 'Y', 'Y')
            print('Trying to import mouse: ', m[''])

            try:
                table.insert1(inputdata)
                print('Mouse: ', m[''], 'imported succesfully')
            except:
                raise ValueError('Failed to import mouse: ', m[''])

    def populate_sessions_table(self):
        """  Populates the sessions table """
        """# A session is one behavioural experiment performed on one mouse on one day
            uid: smallint     # unique number that defines each session
            name: varchar(128)  # unique name that defines each session - YYMMDD_MOUSEID
            ---
            -> Mice                # mouse used for the experiment
            date: date             # date in the YYYY-MM-DD format
            num_recordings: smallint   # number of recordings performed within that session
            experiment_name: varchar(128)  # name of the experiment the session is part of 
            experimenter: varchar(128)      # name of the person performing the experiment
        """

        table = self.sessions
        mice = self.mice.fetch(as_dict=True)
        loaded_excel = pyexcel.get_records(file_name=self.exp_records)

        for session in loaded_excel:
            mouse_id = session['MouseID']
            for mouse in mice:
                idd = mouse['mouse_id']
                original_idd = mouse['mouse_id']
                idd = idd.replace('_', '')
                idd = idd.replace('.', '')
                if idd.lower() == mouse_id.lower():
                    break

            session_name = '{}_{}'.format(session['Date'], session['MouseID'])
            session_date = '20'+str(session['Date'])

            session_data = dict(
                uid = str(session['Sess.ID']),
                name=session_name,
                mouse_id=original_idd,
                date=session_date,
                num_recordings = 0,
                experiment_name=session['Experiment'],
                experimenter='Federico'
            )
            try:
                table.insert1(session_data)
            except:
                raise ValueError('Failed to add session {} to Sessions table'.format(session_name))

    def populate_recordings_table(self):
        """ Populate the Recordings table """
        """
            # Within one session one may perform several recordings. Each recording has its own video and metadata files
            recording_uid: varchar(128)   # uniquely identifying name for each recording YYMMDD_MOUSEID_RECNUM
            ---
            -> Sessions
            rec_num: smallint       # recording number within that session
            video_file_path: varchar(128) # path to the file storing the video data
            video_format: enum('tdms', 'avi', 'mp4')  # format in which the video was recorded
            converted_video_file_path: varchar(128)  # if video was recorded in.tdms and converted to video,where is the video stored
            metadata_file_path: varchar(128) # path to the .tdms file storing the metadata
        """
        print('Populating Recordings Table')
        sessions = self.sessions.fetch(as_dict=True)
        table = self.recordings

        for session in tqdm(sessions):
            print('Getting recordings for session: ', session['uid'], ' - ', session['name'])
            # get video and metadata files
            videos = sorted([f for f in os.listdir(self.raw_video_folder)
                             if session['name'].lower() in f.lower() and 'test' not in f])
            metadatas = sorted([f for f in os.listdir(self.raw_metadata_folder)
                                if session['name'].lower() in f.lower() and 'test' not in f])

            if not videos or not metadatas:
                if not videos and not metadatas: continue
                print('couldnt find files for session: ', session['name'])
            else:
                if len(videos) != len(metadatas):
                    raise ValueError('Something went wront while trying to get the files')

                num_recs = len(videos)
                print(' ... found {} recs'.format(num_recs))

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
                        raise ValueError('Something went wrong while getting recording number within the session')

                    rec_name = session['name']+'_'+str(recnum)
                    format = vid.split('.')[-1]
                    converted = 'nan'

                    # Get deeplabcut data
                    posefile = [os.path.join(self.tracked_data_folder, f) for f in os.listdir(self.tracked_data_folder)
                                if rec_name in f]
                    if not posefile:
                        posefile = [os.path.join(self.tracked_data_folder, f) for f in os.listdir(self.tracked_data_folder)
                                   if session['name'] in f]

                    if len(posefile) != 1:
                        raise ValueError('Failed to load pose data, found {} files'.format(len(posefile)))
                    else: posefile = posefile[0]

                    # pose_data = pd.read_hdf(posefile)

                    # insert recording in table
                    data_to_input = dict(
                        recording_uid=rec_name,
                        uid=session['uid'],
                        name=session['name'],
                        rec_num=rec_num,
                        video_file_path=os.path.join(self.raw_video_folder, vid),
                        video_format=format,
                        converted_video_file_path=converted,
                        metadata_file_path=os.path.join(self.raw_metadata_folder, met),
                        pose_data=posefile
                    )

                    self.insert_entry_in_table(rec_name, 'recording_uid', data_to_input, table, overwrite=False)

    def populate_trials_table(self):
        """# Metadata of each trial (e.g. stim type and frame of onset)
        -> Recordings  --> recording_uid: varchar(128)
        uid: varchar(128)  # uniquely identifuing ID for each trial YYMMDD_MOUSEID_RECNUM_TRIALNUM
        ---
        stim_type: varchar(128)
        stim_start: int   # number of frame at start of stim
        stim_duration: int   # duration in frames
        """
        print('Populating Trials Table')
        recordings = self.recordings.fetch(as_dict=True)
        table = self.trials

        for rec in tqdm(recordings):
            print('processing recording: ', rec['recording_uid'])
            trial_num = 0
            stims = self.load_stimuli_from_tdms(rec['metadata_file_path'])
            for stim_type, stims_frames in stims.items():
                if not stims_frames: continue
                for stim in stims_frames:
                    name = rec['recording_uid'] + '_' + str(trial_num)
                    if stim_type == 'visual': dur = 5*30  # TODO this stuff is hardocoded
                    else: dur = 9*30
                    warnings.warn('Hardcoded variables: stim duration and video fps')

                    data_to_input = dict(recording_uid=rec['recording_uid'],
                                         uid=name,
                                         stim_type=stim_type,
                                         stim_start=stim,
                                         stim_duration=dur)

                    self.insert_entry_in_table(name, 'uid', data_to_input, table, overwrite=False)

    @staticmethod
    def insert_entry_in_table(dataname, checktag, data, table, overwrite=False):
        try:
            table.insert1(data)
            print('     ... inserted {} in table'.format(dataname))
        except:
            if dataname in list(table.fetch(checktag)):
                if overwrite:
                    q = input('Recording entry already in table.\nDo you wish to overwrite? [Y/N]')
                    if q.lower() == 'y':
                        raise ValueError('Feature not implemented yet, overwriting')
                    else:
                        return
            else:
                raise ValueError('Failed to add data entry {} to {} table'.format(dataname, table.full_table_name[-1]))

    def create_trials_clips(self, BehavRec=None, folder=None, prestim=10, poststim=20):
        """
        This function creates small mp4 videos for each trial and saves them.
        It can work on a single BehaviouralRecording or on a whole folder

        :param BehavRec recording to work on, if None work on a whole folder
        :param folder  path to a folder containing videos to be processed, if None self.raw_video_folder
        """
        if BehavRec:
            raise ValueError('Feature not implemented yet: get trial clips for BehaviourRecording')
        else:
            if folder is None:
                video_fld = self.raw_video_folder
                metadata_fld = self.raw_metadata_folder
            else:
                raise ValueError('Feature not implemented yet: get trial clips for custom folder')

        # parameters to draw on frame
        border_size = 20
        color_on = [128, 128, 128]
        color_off = [0, 0, 0]
        curr_color = color_off

        # LOOP OVER EACH VIDEO FILE IN FOLDER
        metadata_files = os.listdir(metadata_fld)
        for v in os.listdir(video_fld):
            print('\n\n\nProcessing: ', v)
            if os.path.getsize(os.path.join(self.raw_video_folder, v)) == 0: continue  # skip if video file is empty

            if 'tdms' in v:  # TODO implemente tdms --> avi conversion
                raise ValueError('Feature not implemented yet: get trial clips from .tdms video')
            else:
                name = os.path.splitext(v)[0]

                # Check if already processed
                processed = [f for f in os.listdir(self.trials_clips) if name in f]
                if processed: continue

                # Load metadata
                tdms_file = [f for f in metadata_files if name == f.split('.')[0]]
                if len(tdms_file)>1: raise ValueError('Could not disambiguate video --> tdms relationship')
                elif not tdms_file:     # Try a couple of things to rescue this error
                    tdms_file = [f for f in metadata_files if name.upper() == f.split('.')[0]]
                    if not tdms_file:
                        tdms_file = [f for f in metadata_files if name.lower() == f.split('.')[0]]
                    if not tdms_file: # give up
                        raise ValueError('Didnt find a tdms file')
                else:
                    # Stimuli frames
                    stimuli = self.load_stimuli_from_tdms(os.path.join(metadata_fld, tdms_file[0]))

                    # Open opencv cap reader and extract video metrics
                    cap = cv2.VideoCapture(os.path.join(self.raw_video_folder, v))
                    if not cap.isOpened():
                        print('Could not process this one')
                        raise ValueError('Could not load video file')

                    fps = cap.get(cv2.CAP_PROP_FPS)
                    window = (int(prestim*fps), int(poststim*fps))
                    clip_number_of_frames = int(window[1]+window[0])

                    # Loop over stims
                    for stim_type, stims in stimuli.items():
                        if stim_type == 'audio':
                            stim_end = window[0] + 9 * fps
                        else:
                            stim_end = window[0] + 5 * fps

                        for stim in stims:
                            width, height = int(cap.get(3)), int(cap.get(4))

                            frame_n = stim-window[0]
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n-1)

                            video_path = os.path.join(self.trials_clips,
                                                      name + '_{}-{}'.format(stim_type, stim) + '.mp4')
                            print('\n\nSaving Clip in: ', video_path)
                            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                            videowriter = cv2.VideoWriter(video_path, fourcc, fps, (width + (border_size * 2),
                                                                                    height + (border_size * 2)), False)

                            for frame_counter in tqdm(range(clip_number_of_frames)):
                                ret, frame = cap.read()
                                if not ret:
                                    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                    if frame_counter + frame_n-1 == tot_frames: break
                                    else:
                                        raise ValueError('Something went wrong when opening next frame: {} of {}'.
                                                         format(frame_counter, tot_frames))

                                # Prep to display stim on
                                if frame_counter < window[0]:
                                    sign = ''
                                    curr_color = color_off
                                else:
                                    sign = '+'
                                    if frame_counter > stim_end: curr_color = color_off
                                    else:
                                        if frame_counter % 15 == 0:
                                            if curr_color == color_off: curr_color = color_on
                                            else: curr_color = color_off

                                # Make frame
                                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                bordered_gray = cv2.copyMakeBorder(gray, border_size, border_size, border_size, border_size,
                                                           cv2.BORDER_CONSTANT, value=curr_color)

                                frame_time = (frame_counter - window[0]) / fps
                                frame_time = str(round(.2 * round(frame_time / .2), 1)) + '0' * (abs(frame_time) < 10)
                                cv2.putText(bordered_gray, sign + str(frame_time) + 's', (width - 110, height + 10), 0, 1,
                                            (180, 180, 180), thickness=2)

                                # Save to file
                                videowriter.write(bordered_gray)

                                frame_counter += 1
                            videowriter.release()



if __name__ == '__main__':
    p = PopulateDatabase()
    p.display_tables_headings()
    # p.create_trials_clips()
    p.populate_mice_table()
    p.populate_sessions_table()
    p.populated_recordings_table()
    p.populate_trials_table()
    p.display_tables_headings()