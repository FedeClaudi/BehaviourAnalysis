import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project

from database.TablesDefinitionsV4 import *

from Utilities.video_and_plotting.video_editing import *
from Utilities.dbase.stim_times_loader import *
from database.database_fetch import *

import datajoint as dj
dj.config["enable_python_native_blobs"] = True

def disable_pandas_warnings():
    import warnings
    warnings.resetwarnings()  # Maybe somebody else is messing with the warnings system?
    warnings.filterwarnings('ignore')  # Ignore everything
    # ignore everything does not work: ignore specific messages, using regex
    warnings.filterwarnings('ignore', '.*A value is trying to be set on a copy of a slice from a DataFrame.*')
    warnings.filterwarnings('ignore', '.*indexing past lexsort depth may impact performance*')


class PopulateDatabase:
    def __init__(self):
        """
        Collection of methods to populate the datajoint database
        """
        # Hard coded paths to relevant files and folders
        with open('paths.yml', 'r') as f:
            paths = yaml.load(f)

        self.define_paths(paths)
        
        self.define_tables([Mouse(),
                            Session(),
                            MazeComponents(),
                            CCM(),
                            Recording(),
                            Stimuli(),
                            TrackingData(),
                            Explorations(), 
                            Trials(),
                            Homings()
                            ])


    def define_tables(self, tables):
        self.all_tables = {}
        for table in tables:
            name = table.table_name.replace("_", "")
            self.__setattr__(name, table)
            self.all_tables[name] = table


    def define_paths(self, paths):
        self.paths = paths
        self.mice_records = paths['mice_records']
        self.exp_records = paths['exp_records']

        self.raw_data_folder = paths['raw_data_folder']
        self.raw_to_sort = os.path.join(self.raw_data_folder, paths['raw_to_sort'])
        self.raw_metadata_folder = os.path.join(self.raw_data_folder, paths['raw_metadata_folder'])
        self.raw_video_folder = os.path.join(self.raw_data_folder, paths['raw_video_folder'])
        self.raw_pose_folder =  paths['tracked_data_folder']
        self.raw_ai_folder = paths["raw_ai_folder"]

        self.trials_clips = os.path.join(self.raw_data_folder, paths['trials_clips'])
        self.tracked_data_folder = paths['tracked_data_folder']

    """
        ###################################################################################################################
        ###################################################################################################################
        ###################################################################################################################
        ###################################################################################################################
    """

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
                    print('Entry with id: {} already in table'.format(dataname))
            else:
                print(table)
                raise ValueError('Failed to add data entry {}-{} to {} table'.format(checktag, dataname, table.full_table_name))

    def delete_wrong_entries(self):
        # query = (Session & "uid > {}".format(429) & "uid < {}".format(444))
        # query.delete()

        sessions_to_delete = np.arange(429, 444)
        for s in sessions_to_delete:
            (TrackingData & "uid={}".format(s)).delete()

        # a = 1


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

    def show_progress(self, tablename=None):
        print("\n\nDatabase update progress")
        if tablename:
            try:
                print(self.all_tables[tablename].progress())
            except:
                print("Cannot show progress for ", tablename)
        else:
            for table in self.all_tables.values():
                try:
                    progr = table.progress(display=False)
                    completed = progr[1] - progr[0]
                    name = table.table_name[1:] + " "*(15-len(table.table_name[1:]))
                    print(name, "  ---  Completed {} of {} ({}%)".format(completed, progr[1], round(completed/progr[1] * 100, 2)))
                except:
                    pass

    def delete_placeholders_from_stim_table(self):
        (self.stimuli & "duration=-1").delete_quick()


    """
        ###################################################################################################################
        ###################################################################################################################
        ###################################################################################################################
        ###################################################################################################################
    """

    def populate_mice_table(self):
        """ Populates the Mice() table from the database"""
        table = self.mouse
        loaded_excel = pyexcel.get_records(file_name=self.mice_records)

        for m in loaded_excel:
            if not m['']: continue

            mouse_data = dict(
                mouse_id = m[''],
                strain = m['Strain'],
                sex = 'M',
            )
            self.insert_entry_in_table(mouse_data['mouse_id'], 'mouse_id', mouse_data, table)

    def populate_sessions_table(self):
        """  Populates the sessions table """
        mice = self.mouse.fetch(as_dict=True)
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
            self.insert_entry_in_table(session_data['session_name'], 'session_name', session_data, self.session)
        
            # Insert into metadata part table
            part_dat = dict(
                session_name=session_data["session_name"],
                uid=session_data["uid"],
                maze_type= int(session["Maze type"]),
                naive = int(session["Naive"]),
                lights = int(session["Lights"]),
                mouse_id=original_idd,
            )

            self.insert_entry_in_table(part_dat['session_name'], 'session_name', part_dat, self.session.Metadata)

            # Insert into shelter metadata
            part_dat = dict(
                session_name=session_data["session_name"],
                uid=session_data["uid"],
                shelter= int(session["Shelter"]),
                mouse_id=original_idd,

            )

            self.insert_entry_in_table(part_dat['session_name'], 'session_name', part_dat, self.session.Shelter)


    """
        ###################################################################################################################
        ###################################################################################################################
    """


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
    """
        ###################################################################################################################
        ###################################################################################################################
    """


if __name__ == '__main__':
    disable_pandas_warnings()
    p = PopulateDatabase()

    # print(p)

    errors = []


    # ? drop tables
    # p.remove_table(["recording"])

    # ? Remove stuff from tables
    # p.delete_placeholders_from_stim_table()
    # p.delete_wrong_entries()
        
    # ? These tables population is fast and largely automated
    # p.populate_mice_table()   # ! mice recordings, components... 
    # p.populate_sessions_table()
 
    # p.recording.populate(display_progress=True) 
    # p.recording.make_paths(p) 
    # p.mazecomponents.populate(display_progress=True)  # ? this will require input for new experiments

    # ? This slower and will require some input
    # Before populating CCM you need to have done the tracking and have ran recording.make_paths
    # p.ccm.populate(display_progress=True)  # ! ccm

    # ? this is considerably slower but should be automated
    errors = p.trackingdata.populate(display_progress=True, suppress_errors=False, return_exception_objects =True) # ! tracking data

    # errors = p.stimuli.populate(display_progress=True, suppress_errors=False, return_exception_objects=True) # , max_calls =10)  # ! stimuli
    # p.stimuli.make_metadata() # ? only used for visual stims

    # ? Should be fast but needs the stuff above to be done
    # p.explorations.populate(display_progress=True, suppress_errors=False, return_exception_objects =True)
    # p.trials.populate(display_progress=True, suppress_errors=False, return_exception_objects =True)

    # if errors: raise ValueError([print("\n\n", e) for e in errors])

    # ? Show database content and progress
    # print(p.ccm.tail())
    p.show_progress()





