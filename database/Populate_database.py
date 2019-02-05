import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project

from database.dj_config import start_connection
start_connection()

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
from collections import namedtuple

from database.NewTablesDefinitions import *

from Utilities.video_and_plotting.video_editing import *
from Utilities.stim_times_loader import *


class PopulateDatabase:
    def __init__(self):
        """
        Collection of methods to populate the datajoint database
        """
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
        self.videofiles = VideoFiles()
        self.behaviourstimuli = BehaviourStimuli()
        self.mantisstimuli = MantisStimuli()
        self.videosincomplete = VideosIncomplete()
        self.tracking_data = TrackingData()
        self.commoncoordinatematrices = CommonCoordinateMatrices()
        self.dlcmodels = DLCmodels()
        self.all_tables = dict(mice=self.mice, sessions= self.sessions, experiments=self.experiments,
                                recordings=self.recordings, behaviourstimuli = self.behaviourstimuli,
                                mantisstimuli = self.mantisstimuli, dlcmodels = self.dlcmodels,
                                templates=self.templates, videofiles = self.videofiles, 
                                commoncoordinatematrices=self.commoncoordinatematrices,
                                tracking_data = self.tracking_data, videosincomplete = self.videosincomplete)

    def remove_table(self, tablename):
        """
        removes a single table from the database
        """
        if isinstance(tablename, str):
            tablename = [tablename]
        for table in tablename:
            tb = self.all_tables[table]
            tb.drop()
        sys.exit()


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

    # def remove_test_sessions(self):
    #     # Some sessions on the datalog are test sessions which should not be analysed properly. 
    #     test_sessions_ids = [87, 88]
    #     for id in test_sessions_ids:
    #         try:
    #             print('Deleting...\n\n', ((self.tracking_data & 'uid={}'.format(id))))
    #         except:
    #             print('Could not delte sessions with id ', id)
    #         else:
    #             (self.tracking_data & 'uid={}'.format(id) ).delete()


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
        
        # Remove test sessions
        # p.remove_test_sessions()

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


    def display_videos_incomplete(self):
        fetched = self.videosincomplete.fetch(as_dict=True)
        tot_conversions, tot_dlcs = 0, 0
        for entry in fetched:
            print('Recording: {}-{} - conversion: {} - dlc: {}'.format(entry['uid'], entry['recording_uid'], 
                    entry['conversion_needed'], entry['dlc_needed']))
            if entry['conversion_needed'] == 'true': tot_conversions += 1
            if entry['dlc_needed'] == 'true': tot_dlcs += 1

        print('\n\n In total there are {} incomplete videos of which\n    {} need conversion\n    {} need dlc'.format(
                len(fetched), tot_conversions, tot_dlcs))


    def __str__(self):
        self.__repr__()
        return ''

    def __repr__(self):
        summary = {}
        tabledata = namedtuple('data', 'name numofentries lastentry')
        for name, table in self.all_tables.items():
            if table is None: continue
            fetched = table.fetch()
            df = pd.DataFrame(fetched)
            toprint = tabledata(name, len(fetched), df.tail(1))

            summary[name] = toprint.numofentries

            # print('Table {} has {} entries'.format(toprint.name, toprint.numofentries))
            # print('The last entry in the table is\n ', toprint.lastentry)

        print('\n\nNumber of Entries per table')
        sumdf = (pd.DataFrame.from_dict(summary, orient='index'))
        sumdf.columns = ['NumOfEntries']
        print(sumdf)
        return ''


if __name__ == '__main__':
    p = PopulateDatabase()

    print(p)

    # p.remove_table(['tracking_data',])

    # p.populate_mice_table()
    # p.populate_experiments_table()
    # p.populate_sessions_table()y
    # p.dlcmodels.populate()

    # p.recordings.populate()

    # p.videofiles.populate()

    # p.commoncoordinatematrices.populate()
    # p.templates.populate()


    
    # p.behaviourstimuli.populate()
    # p.mantisstimuli.populate()

    p.tracking_data.populate()


    # p.remove_test_sessions()

    # print(p.sessions)
