from Tables_definitions import *

import datajoint as dj
import os
import pyexcel
import yaml
import moviepy
import cv2
from moviepy.editor import VideoFileClip
from shutil import copyfile
from tqdm import tqdm


from utils.video_editing import *

class PopulateDatabase:
    def __init__(self):
        """
        Collection of methods to populate the different


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
        with open('./data_paths.yml', 'r') as f:
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

    def update_mice_table(self):
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
            try:
                mice.insert1(inputdata)
            except:
                a = 1

    def update_sessions_table(self):
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

                session_data = dict(
                    uid = None,
                    name=str(session['Sess.ID']),
                    mouse_id=original_idd,
                    session_date=session['Date'],
                    num_recordings = None,
                    experiment=session['Experiment'],
                    experimenter='Federico'
                )
                try:
                    sessions.insert1(session_data)
                except:
                    pass


    @staticmethod
    def load_stimuli_from_tdms(tdmspath, software='behaviour'):
        """ Takes the path to a tdms file and attempts to load the stimuli metadata from it, returns a dictionary
         with the stim times """
        # TODO load metadata
        # Try to load a .tdms
        print('\n Loading stimuli time from .tdms: {}'.format(os.path.split(tdmspath)[-1]))
        try:
            tdms = TdmsFile(tdmspath)
        except:
            raise ValueError('Could not load .tdms file')

        if software == 'behaviour':
            stimuli = dict(audio=[], visual=[], digital=[])
            for group in tdms.groups():
                for obj in tdms.group_channels(group):
                    if 'stimulis' in str(obj).lower():
                        for idx in obj.as_dataframe().loc[0].index:
                            if '  ' in idx:
                                framen = int(idx.split('  ')[1].split('-')[0])
                            else:
                                framen = int(idx.split(' ')[2].split('-')[0])

                            if 'visual' in str(obj).lower():
                                stimuli['visual'].append(framen)
                            elif 'audio' in str(obj).lower():
                                stimuli['audio'].append(framen)
                            elif 'digital' in str(obj).lower():
                                stimuli['digital'].append(framen)
                            else:
                                print('                  ... couldnt load stim correctly')
        else:
            raise ValueError('Feature not implemented yet: load stim metdata from Mantis .tdms')
        return stimuli

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
            if os.path.getsize(os.path.join(self.raw_video_folder, v)) == 0: break  # skip if video file is empty

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
                elif not tdms_file:
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

    def sort_behaviour_files(self):
        for fld in os.listdir(self.raw_to_sort):
            for f in os.listdir(os.path.join(self.raw_to_sort, fld)):
                print('sorting ', fld)
                if '.tdms' in f and 'index' not in f:
                    copyfile(os.path.join(self.raw_to_sort, fld, f),
                             os.path.join(self.raw_metadata_folder, f))
                elif '.avi' in f:
                    os.rename(os.path.join(self.raw_to_sort, fld, f),
                              os.path.join(self.raw_to_sort, fld, fld+'.avi'))
                    copyfile(os.path.join(self.raw_to_sort, fld, fld+'.avi'), 
                             os.path.join(self.raw_video_folder, fld+'.avi'))
                else:
                    pass


if __name__ == '__main__':
    p = PopulateDatabase()
    # p.create_trials_clips()
    p.display_tables_headings()


