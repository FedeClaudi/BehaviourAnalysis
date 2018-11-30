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
        Collection of methods to populate the datajoint database
        """
        print("""
        Ready to populate database. Available classes:
                * Mice
                * Sessions
                * Recordings
                * Trials""")

        # Hard coded paths to relevant files and folders
        with open('paths.yml', 'r') as f:
            paths = yaml.load(f)

        self.paths = paths

        self.mice_records = paths['mice_records']
        self.exp_records = paths['exp_records']

        self.raw_data_folder = paths['raw_data_folder']
        self.raw_to_sort = os.path.join(self.raw_data_folder, paths['raw_to_sort'])
        self.raw_metadata_folder = os.path.join(self.raw_data_folder, paths['raw_metadata_folder'])
        self.raw_video_folder = os.path.join(self.raw_data_folder, paths['raw_video_folder'])

        self.trials_clips = os.path.join(self.raw_data_folder, paths['trials_clips'])
        self.tracked_data_folder = paths['tracked_data_folder']

        # Load tables definitions
        self.mice = Mice()
        self.experiments = Experiments()
        self.templates = Templates()
        self.sessions = Sessions()
        self.recordings = Recordings()
        self.stimuli = Stimuli()
        self.tracking_data = TrackingData()
        self.all_tables = dict(mice=self.mice, sessions= self.sessions, experiments=self.experiments, recordings=self.recordings,
                               stimuli=self.stimuli, tracking_data = self.tracking_data, templates=self.templates)

    def display_tables_headings(self):
        """
        prints out each table with the first N entries
        """
        for name, table in self.all_tables.items():
            print('\n\nTable definition for {}:\n{}'.format(name, table))

    def reset_database(self):
        """
        tries to remove all tables from the database
        """
        print('ATTENTION: this might result in loss of data!!!')
        q = input('Continue ? [Y/N]')
        if not q.lower() == 'y':
            return
        else:
            [table.drop() for table in self.all_tables.values()]

    def remove_table(self, tablename):
        """
        removes a single table from the database
        """
        tb = self.all_tables[tablename]
        tb.drop()

    def populate_mice_table(self):
        """ Populates the Mice() table from the database"""
        table = self.mice
        loaded_excel = pyexcel.get_records(file_name=self.mice_records)

        for m in loaded_excel:
            if not m['']: continue

            mouse_data = dict(
                mouse_id = m[''],
                strain = m['Strain'],
                dob = m['DOB'].strip(),
                sex = 'M',
                single_housed = 'Y',
                enriched_cage = 'Y'
            )
            self.insert_entry_in_table(mouse_data['mouse_id'], 'mouse_id', mouse_data, table)

    def populate_experiments_table(self):
        table = self.experiments

        exp_names = [d for d in os.listdir(self.paths['maze_templates']) if d != 'ignored']

        for exp in exp_names:
            data_to_input = dict(
                experiment_name=exp,
                templates_folder = os.path.join(self.paths['maze_templates'], exp)
            )
            self.insert_entry_in_table(data_to_input['experiment_name'], 'experiment_name', data_to_input, table)

    def populate_sessions_table(self):
        """  Populates the sessions table """
        table = self.sessions
        mice = self.mice.fetch(as_dict=True)
        loaded_excel = pyexcel.get_records(file_name=self.exp_records)

        for session in loaded_excel:
            # Get mouse name
            mouse_id = session['MouseID']
            for mouse in mice:
                idd = mouse['mouse_id']
                original_idd = mouse['mouse_id']
                idd = idd.replace('_', '')
                idd = idd.replace('.', '')
                if idd.lower() == mouse_id.lower():
                    break

            # Get session name
            session_name = '{}_{}'.format(session['Date'], session['MouseID'])
            session_date = '20'+str(session['Date'])

            # Get experiment name
            experiment_name = session['Experiment']

            # Insert into table
            session_data = dict(
                uid = str(session['Sess.ID']),
                session_name=session_name,
                mouse_id=original_idd,
                date=session_date,
                experiment_name = experiment_name
            )
            self.insert_entry_in_table(session_data['session_name'], 'session_name', session_data, table)

    def populate_recordings_table(self):
        """ Populate the Recordings table """
        print('Populating Recordings Table')
        sessions = self.sessions.fetch(as_dict=True)
        table = self.recordings

        for session in tqdm(sessions):
            print('Getting recordings for session: ', session['uid'], ' - ', session['session_name'])
            # get video and metadata files
            videos = sorted([f for f in os.listdir(self.raw_video_folder)
                             if session['session_name'].lower() in f.lower() and 'test' not in f
                             and '.h5' not in f and '.pickle' not in f])
            metadatas = sorted([f for f in os.listdir(self.raw_metadata_folder)
                                if session['session_name'].lower() in f.lower() and 'test' not in f and '.tdms' in f])

            if not videos or not metadatas:
                if not videos and not metadatas: continue
                print('couldnt find files for session: ', session['session_name'])
                raise FileNotFoundError('dang')
            else:
                if len(videos) != len(metadatas):
                    print('Found {} videos files: {}'.format(len(videos), videos))
                    print('Found {} metadatas files: {}'.format(len(metadatas), metadatas))
                    raise ValueError('Something went wront wrong trying to get the files')

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

                    rec_name = session['session_name']+'_'+str(recnum)
                    format = vid.split('.')[-1]
                    converted = 'nan'

                    # Get deeplabcut data
                    print(rec_name)
                    posefile = [os.path.join(self.tracked_data_folder, f) for f in os.listdir(self.tracked_data_folder)
                                if rec_name == os.path.splitext(f)[0].split('Deep')[0] and '.pickle' not in f]
                    if not posefile:
                        print('didnt find pose file, trying harder')
                        posefile = [os.path.join(self.tracked_data_folder, f) for f in os.listdir(self.tracked_data_folder)
                                   if session['session_name'] in f and '.pickle' not in f]

                    if len(posefile) != 1:
                        if rec_name in table.fetch('recording_uid'): continue  # no need to worry about it

                        print("\n\n\nCould not find pose data for recording {}".format(rec_name))
                        if posefile:
                            print('Found these possible matches: ')
                            [print('\n[{}] - {}'.format(i,f)) for i,f in enumerate(posefile)]
                            yn = input("\nPlease select file [or type 'y' if none matches and you wish to continue anyways, n otherwise]:  int/y/n  ")
                        else:
                            yn = input('\nNo .h5 file found, continue anyways??  y/n  ')
                        if yn == 'n': 
                            yn = input('\nDo you want to instert this recording withouth a pose file??  y/n  ')
                            if yn == 'y':
                                posefile = 'nan'
                            else:
                                raise ValueError('Failed to load pose data, found {} files for recording --- \n         {}\n{}'.format(len(posefile), 
                                                                                                                            rec_name, posefile))
                        elif yn == 'y':
                            continue
                        else:
                            try:
                                sel = int(yn)
                                posefile = posefile[sel]
                            except:
                                raise ValueError('Failed to load pose data, found {} files for recording --- \n         {}\n{}'.format(len(posefile), 
                                                                                                                            rec_name, posefile))

                    # insert recording in table
                    data_to_input = dict(
                        recording_uid=rec_name,
                        uid=session['uid'],
                        session_name=session['session_name'],
                        rec_num=rec_num,
                        video_file_path=os.path.join(self.raw_video_folder, vid),
                        video_format=format,
                        converted_video_file_path=converted,
                        metadata_file_path=os.path.join(self.raw_metadata_folder, met),
                        pose_data=posefile
                    )

                    self.insert_entry_in_table(rec_name, 'recording_uid', data_to_input, table, overwrite=False)

    def populate_templates_table(self):
        self.templates.populate()

    def populate_stimuli_table(self):
        self.stimuli.populate()


    @staticmethod
    def insert_entry_in_table(dataname, checktag, data, table, overwrite=False):
        """
        dataname: value of indentifying key for entry in table
        checktag: name of the identifying key ['those before the --- in the table declaration']
        data: entry to be inserted into the table
        table: database table
        """
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
                    print('Entry with id: {} already in table'.format(dataname))
            else:
                print(table)
                raise ValueError('Failed to add data entry {}-{} to {} table'.format(checktag, dataname, table.full_table_name))


if __name__ == '__main__':
    p = PopulateDatabase()

    # print(p.mice)
    # print(p.experiments)

    p.remove_table('templates')
    sys.exit()

    # p.populate_mice_table()
    # p.populate_experiments_table()
    # p.populate_sessions_table()
    # p.populate_recordings_table()
    # p.populate_trials_table()
    # p.display_tables_headings()