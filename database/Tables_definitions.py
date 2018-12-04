import sys
sys.path.append('./')  

import datajoint as dj
from database.dj_config import start_connection
dbname = start_connection()
schema = dj.schema(dbname, locals())
from nptdms import TdmsFile

import pandas as pd
import os
from collections import namedtuple
import numpy as np

from Utilities.file_io.files_load_save import load_yaml

@schema
class Mice(dj.Manual):
    definition = """
      # Mouse table lists all the mice used and the relevant attributes
      mouse_id: varchar(128)                        # unique mouse id
      ---
      strain:   varchar(128)                        # genetic strain
      dob: varchar(128)                             # mouse date of birth 
      sex: enum('M', 'F', 'U')                      # sex of mouse - Male, Female, or Unknown/Unclassified
      single_housed: enum('Y', 'N')                 # single housed or group caged
      enriched_cage: enum('Y', 'N')                 # presence of wheel or other stuff in the cage
      """

@schema
class Experiments(dj.Manual):
    definition = """
    # Name of the experiments and location of components templates
    experiment_name: varchar(128)
    ---
    templates_folder: varchar(256)
    """

@schema
class Sessions(dj.Manual):
    definition = """
    # A session is one behavioural experiment performed on one mouse on one day
    uid: smallint     # unique number that defines each session
    session_name: varchar(128)  # unique name that defines each session - YYMMDD_MOUSEID
    ---
    -> Mice
    date: date             # date in the YYYY-MM-DD format
    experiment_name: varchar(128)  # name of the experiment the session is part of 
    -> Experiments      # name of the person performing the experiment
    """

@schema
class Recordings(dj.Imported):
    definition = """
    # Within one session one may perform several recordings. Each recording has its own video and metadata files
    recording_uid: varchar(128)   # uniquely identifying name for each recording YYMMDD_MOUSEID_RECNUM
    -> Sessions
    ---
    """

    class VideoFiles(dj.Part):
        definition = """
            # stores paths to video files
            -> Recordings
            ---
            overview: varchar(256)          # overview camera
            threat: varchar(256)            # threat camera
            top_mirror: varchar(256)        # top mirror view
            side_mirror: varchar(256)       # side mirror view
            """

    class ConvertedVideoFiles(dj.Part):
        definition = """
            # stores paths to converted videos (from tdms to mp4), if no conversion is made then same as VideoFiles
            -> Recordings
            ---
            overview: varchar(256)          # overview camera
            threat: varchar(256)            # threat camera
            top_mirror: varchar(256)        # top mirror view
            side_mirror: varchar(256)       # side mirror view
            """

    class PoseFiles(dj.Part):
        definition = """
            # stores paths to converted videos (from tdms to mp4), if no conversion is made then same as VideoFiles
            -> Recordings
            ---
            overview: varchar(256)          # overview camera
            threat: varchar(256)            # threat camera
            top_mirror: varchar(256)        # top mirror view
            side_mirror: varchar(256)       # side mirror view
            """

    class MetadataFiles(dj.Part):
        definition = """
            # stores paths to metadta .tdms files with video recording metadata
            -> Recordings
            ---
            overview: varchar(256)          # overview camera
            threat: varchar(256)            # threat camera
            """

    class AnalogInputs(dj.Part):
        definition = """
            # Stores data from relevant AI channels recorded with NI board
            -> Recordings
            ---
            overview_camera_triggers: longblob      # Frame triggers signals efferent copy
            threat_camera_triggers: longblob
            speaker_signal: longblob                # HIGH when ultrasound being produced
        """


    def make(self, session):
        """ Populate the Recordings table """
        
        paths = load_yaml('paths.yml')
        raw_video_folder = os.path.join(paths['raw_data_folder'], paths['raw_video_folder'])
        raw_metadata_folder = os.path.join(paths['raw_data_folder'], paths['raw_metadata_folder'])
        tracked_data_folder = paths['tracked_data_folder']

        # get video and metadata files
        videos = sorted([f for f in os.listdir(raw_video_folder)
                            if session['session_name'].lower() in f.lower() and 'test' not in f
                            and '.h5' not in f and '.pickle' not in f])
        metadatas = sorted([f for f in os.listdir(raw_metadata_folder)
                            if session['session_name'].lower() in f.lower() and 'test' not in f and '.tdms' in f])

        # Make sure we got the correct number of files, otherwise ask for user input
        if not videos or not metadatas:
            if not videos and not metadatas: return
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
                posefile = [os.path.join(tracked_data_folder, f) for f in os.listdir(tracked_data_folder)
                            if rec_name == os.path.splitext(f)[0].split('Deep')[0] and '.pickle' not in f]
                if not posefile:
                    print('didnt find pose file, trying harder')
                    posefile = [os.path.join(tracked_data_folder, f) for f in os.listdir(tracked_data_folder)
                                if session['session_name'] in f and '.pickle' not in f]

                if len(posefile) != 1:
                    if rec_name in self.fetch('recording_uid'): continue  # no need to worry about it

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

                # insert recording in main table
                session['recording_uid'] = rec_name
                self.insert1(session)

                # Insert stuff into part tables
                # prep
                cameras = ['overview', 'threat', 'top_mirror', 'side_mirror']

                videofiles = {c:'nan' for c in cameras}
                videofiles['overview'] = os.path.join(raw_video_folder, vid)

                convertedvideofiles = videofiles.copy()

                metadatafiles = {c: 'nan' for c in cameras if c not in ['top_mirror', 'side_mirror']}
                metadatafiles['overview'] = os.path.join(raw_metadata_folder, met)

                posefiles = {c: 'nan' for c in cameras}
                posefiles['overview'] = posefile

                all_dics = [videofiles, convertedvideofiles, metadatafiles, posefiles]
                for d in all_dics:
                    d['recording_uid']=rec_name 
                    d['uid'] = session['uid']
                    d['session_name'] = session['session_name']

                # actually insert
                print(videofiles, convertedvideofiles, metadatafiles)
                Recordings.VideoFiles.insert1(videofiles)
                Recordings.ConvertedVideoFiles.insert1(convertedvideofiles)
                Recordings.PoseFiles.insert1(posefiles)
                Recordings.MetadataFiles.insert1(metadatafiles)

@schema
class Templates(dj.Imported):
    definition = """
    # stores the position of each maze template for one experiment
    -> Recordings
    ---
    s: longblob  # Shelter platform template position
    t: longblob  # Threat platform
    p1: longblob  # Other platforms
    p2: longblob
    p3: longblob
    p4: longblob
    p5: longblob
    p6: longblob
    b1: longblob  # Bridges
    b2: longblob
    b3: longblob
    b4: longblob
    b5: longblob
    b6: longblob
    b7: longblob
    b8: longblob
    b9: longblob
    b10: longblob
    b11: longblob
    b12: longblob
    b13: longblob
    b14: longblob
    """

    def _make_tuples(self, key):
        from Processing.rois_toolbox.get_maze_components import get_rois_from_templates

        # Get all possible components name
        nplatf, nbridges = 6, 14
        platforms = ['p'+str(i) for i in range(1, nplatf + 1)]
        bridges = ['b'+str(i) for i in range(1, nbridges + 1)]
        all_components = ['s', 't']
        all_components.extend(platforms)
        all_components.extend(bridges)

        # Get entries from related tables
        session = [s for s in Sessions.fetch() if s['session_name']==key['session_name']][0]
        recording = [r for r in Recordings.fetch() if r['recording_uid'] == key['recording_uid']][0]
        experiment = [e for e in Experiments.fetch() if e['experiment_name']==session['experiment_name']]
        if not experiment or not isinstance(experiment, list):
            print(Experiments.fetch('experiment_name'))
            raise ValueError('Could not find match for experiment: ', session['experiment_name'])
        else:
            experiment = experiment[0]

        videofile = [v for v in Recordings.VideoFiles.fetch() if v['recording_uid'] == key['recording_uid']]
        if not videofile or not isinstance(videofile, list):
            return
            raise FileNotFoundError(' could not find videofile, found: {} for recording {} in session {}'.format(
                                    videofile, recording, session))
        else:
            videofile = videofile[0]

        # Get matched components for recording
        templates_fld = experiment['templates_folder']
        video = videofile['overview']
        matched = get_rois_from_templates(session, video, templates_fld)

        # Prepare data to insert into the table
        data_to_input =  {(n if n in matched.keys() else n):(matched[n] if n in matched.keys() else 0)
                           for n in all_components}

        for k,v in data_to_input.items():
            key[k] = v

        # Insert
        self.insert1(key)

@schema
class Stimuli(dj.Computed):
    definition = """
    # Metadata of each trial (e.g. stim type and frame of onset)
    -> Recordings
    stimulus_uid: varchar(128)  # uniquely identifuing ID for each trial YYMMDD_MOUSEID_RECNUM_TRIALNUM
    ---
    stim_type: varchar(128)
    stim_start: int   # number of frame at start of stim
    stim_duration: int   # duration in frames
    stim_metadata: longblob  # list of other stuff ? 
    """

    def make(self, key):
        # TODO Mantis
        # TODO more metadata

        videofile = [v for v in Recordings.MetadataFiles.fetch() if v['recording_uid'] == key['recording_uid']]
        if not videofile or not isinstance(videofile, list):
            raise ValueError('Could not fetch path')
        else:
            videofile = videofile[0]

        tdmspath = videofile['overview']
        recording_uid = key['recording_uid']

        # Try to load a .tdms
        try:
            print('           ... loading metadata from .tdms: {}'.format(os.path.split(tdmspath)[-1]))
            tdms = TdmsFile(tdmspath)
        except FileNotFoundError:
            raise FileNotFoundError(' Could not load tdms file')

        # Get all stimuli in .tdms
        stimuli = {}
        for group in tdms.groups():
            for obj in tdms.group_channels(group):
                for idx in obj.as_dataframe().loc[0].index:
                    if 'stimulis' in str(obj).lower():
                        # Get stim type
                        if 'visual' in str(obj).lower():
                            stim_type = 'visual'
                            stim_duration = 5 * 30  # ! <-
                        elif 'audio' in str(obj).lower():
                            stim_type = 'audio'
                            stim_duration = 9 * 30  # ! <-

                        # Get stim frame
                        if '  ' in idx:
                            framen = int(idx.split('  ')[1].split('-')[0])
                        else:
                            framen = int(idx.split(' ')[2].split('-')[0])

                        stimuli[str(framen)] = stim_type


        # Insert entries in table
        if len(list(stimuli.keys())) == 0:
            # Insert void entry so that the empty tdms will not be loaded
            # again the next time the make() attribute is called
            data_to_input = dict(
                    recording_uid = recording_uid,
                    uid = key['uid'],
                    session_name = key['session_name'],
                    stimulus_uid=recording_uid + '_{}'.format(-1),
                    stim_type = 'visual',
                    stim_start = -1,
                    stim_duration = 0,
                    stim_metadata = [0])
            print(' inserting empty values', data_to_input)
            self.insert1(data_to_input)
        else:
            for i, k in enumerate(sorted(stimuli.keys())):
                stim = stimuli[k]

                data_to_input = dict(
                    recording_uid = recording_uid,
                    uid = key['uid'],
                    session_name = key['session_name'],
                    stimulus_uid=recording_uid + '_{}'.format(i),
                    stim_type = stim_type,
                    stim_start = int(k),
                    stim_duration = stim_duration,
                    stim_metadata = [0]  # ! <- 
                )
                print(data_to_input)
                self.insert1(data_to_input)

@schema
class TrackingData(dj.Computed):
    definition = """
    # Stores the DLC recording data for one Recording entry
    -> Recordings
    """

    def define_attrs(self):
        self.attributes = dict(
            LeftEar = self.LeftEar,
            LeftEye = self.LeftEye,
            Snout = self.Snout,
            RightEye = self.RightEye,
            RightEar = self.RightEar,
            Neck = self.Neck,
            RightShoulder = self.RightShoulder,
            RightHip = self.RightHip,
            TailBase = self.TailBase,
            Tail2 = self.Tail2,
            Tail3 = self.Tail3,
            LeftHip = self.LeftHip,
            LeftShoulder = self.LeftShoulder,
            Body = self.Body,
        )

    def getallattr(self):
        self.define_attrs()
        return self.attributes

    def getattr(self, attrname):
        self.define_attrs()
        try:
            return self.attributes[attrname]
        except:
            if attrname == 'RightShould':
                return self.attributes['RightShoulder']
            else:
                raise ValueError('Could not find attribute ', attrname)
                
    # Docs on parts table 
    # https://docs.datajoint.io/computation/Part-tables.html

    class LeftEar(dj.Part):
        """
            attributes = dict(
                        LeftEar = self.LeftEar,
                        LeftEye = self.LeftEye,
                        Snout = self.Snout,
                        RightEye = self.RightEye,
                        RightEar = self.RightEar,
                        Neck = self.Neck,
                        RightShoulder = self.RightShoulder,
                        RightHip = self.RightHip,
                        TailBase = self.TailBase,
                        Tail2 = self.Tail2,
                        Tail3 = self.Tail3,
                        LeftHip = self.LeftHip,
                        LeftShoulder = self.LeftShoulder,
                        Body = self.Body,
                    )
        """
        definition = """
            -> TrackingData
            ---
            overview: longblob     # pose data extracted from overview camera
            threat: longblob       # pose data extracted from threat camear (main view)
            top_mirror: longblob   # pose data extracted from threat camera top mirror
            side_mirror: longblob  # pose data extracted from threat camera side mirror
        """

    class LeftEye(dj.Part):
        definition = """
            -> TrackingData
            ---
            overview: longblob     # pose data extracted from overview camera
            threat: longblob       # pose data extracted from threat camear (main view)
            top_mirror: longblob   # pose data extracted from threat camera top mirror
            side_mirror: longblob  # pose data extracted from threat camera side mirror
        """

    class Snout(dj.Part):
        definition = """
            -> TrackingData
            ---
            overview: longblob     # pose data extracted from overview camera
            threat: longblob       # pose data extracted from threat camear (main view)
            top_mirror: longblob   # pose data extracted from threat camera top mirror
            side_mirror: longblob  # pose data extracted from threat camera side mirror
        """

    class RightEye(dj.Part):
        definition = """
            -> TrackingData
            ---
            overview: longblob     # pose data extracted from overview camera
            threat: longblob       # pose data extracted from threat camear (main view)
            top_mirror: longblob   # pose data extracted from threat camera top mirror
            side_mirror: longblob  # pose data extracted from threat camera side mirror
        """

    class RightEar(dj.Part):
        definition = """
            -> TrackingData
            ---
            overview: longblob     # pose data extracted from overview camera
            threat: longblob       # pose data extracted from threat camear (main view)
            top_mirror: longblob   # pose data extracted from threat camera top mirror
            side_mirror: longblob  # pose data extracted from threat camera side mirror
        """

    class Neck(dj.Part):
        definition = """
            -> TrackingData
            ---
            overview: longblob     # pose data extracted from overview camera
            threat: longblob       # pose data extracted from threat camear (main view)
            top_mirror: longblob   # pose data extracted from threat camera top mirror
            side_mirror: longblob  # pose data extracted from threat camera side mirror
        """

    class RightShoulder(dj.Part):
        definition = """
            -> TrackingData
            ---
            overview: longblob     # pose data extracted from overview camera
            threat: longblob       # pose data extracted from threat camear (main view)
            top_mirror: longblob   # pose data extracted from threat camera top mirror
            side_mirror: longblob  # pose data extracted from threat camera side mirror
        """

    class RightHip(dj.Part):
        definition = """
            -> TrackingData
            ---
            overview: longblob     # pose data extracted from overview camera
            threat: longblob       # pose data extracted from threat camear (main view)
            top_mirror: longblob   # pose data extracted from threat camera top mirror
            side_mirror: longblob  # pose data extracted from threat camera side mirror
        """

    class TailBase(dj.Part):
        definition = """
            -> TrackingData
            ---
            overview: longblob     # pose data extracted from overview camera
            threat: longblob       # pose data extracted from threat camear (main view)
            top_mirror: longblob   # pose data extracted from threat camera top mirror
            side_mirror: longblob  # pose data extracted from threat camera side mirror
        """

    class Tail2(dj.Part):
        definition = """
            -> TrackingData
            ---
            overview: longblob     # pose data extracted from overview camera
            threat: longblob       # pose data extracted from threat camear (main view)
            top_mirror: longblob   # pose data extracted from threat camera top mirror
            side_mirror: longblob  # pose data extracted from threat camera side mirror
        """
    
    class Tail3(dj.Part):
        definition = """
            -> TrackingData
            ---
            overview: longblob     # pose data extracted from overview camera
            threat: longblob       # pose data extracted from threat camear (main view)
            top_mirror: longblob   # pose data extracted from threat camera top mirror
            side_mirror: longblob  # pose data extracted from threat camera side mirror
        """

    class LeftHip(dj.Part):
        definition = """
            -> TrackingData
            ---
            overview: longblob     # pose data extracted from overview camera
            threat: longblob       # pose data extracted from threat camear (main view)
            top_mirror: longblob   # pose data extracted from threat camera top mirror
            side_mirror: longblob  # pose data extracted from threat camera side mirror
        """

    class LeftShoulder(dj.Part):
        definition = """
            -> TrackingData
            ---
            overview: longblob     # pose data extracted from overview camera
            threat: longblob       # pose data extracted from threat camear (main view)
            top_mirror: longblob   # pose data extracted from threat camera top mirror
            side_mirror: longblob  # pose data extracted from threat camera side mirror
        """

    class Body(dj.Part):
        definition = """
            -> TrackingData
            ---
            overview: longblob     # pose data extracted from overview camera
            threat: longblob       # pose data extracted from threat camear (main view)
            top_mirror: longblob   # pose data extracted from threat camera top mirror
            side_mirror: longblob  # pose data extracted from threat camera side mirror
        """

    def make(self, key):
        print('\n\nPopulating Tracking data\n', key)
        rec_name = key['recording_uid']
        pose_files = [ff for ff in Recordings.PoseFiles.fetch() if ff['recording_uid']==rec_name][0]

        # initialise empty dict of dict
        allbp = {}
        cameras = ['overview', 'threat', 'top_mirror', 'side_mirror']
        bodyparts = ['left_ear', 'left_eye', 'snout', 'right_eye', 'right_ear', 'neck', 'right_should',
                     'right_hip', 'tail_base', 'tail_2', 'tail_3', 'left_hip', 'left_shoulder', 'body']
        for bp in bodyparts:
            allbp[bp] = {cam:None for cam in cameras}

        # now fill that dict with dem data
        for cam in cameras:
            pfile = pose_files[cam]
            if pfile == 'nan': continue

            try:
                pose = pd.read_hdf(pfile)  # ? load pose
            except FileNotFoundError:
                print('Could not open file: ', pfile)
                print(pose_files)
                raise FileExistsError()
            
            first_frame = pose.iloc[0]

            bodyparts = first_frame.index.levels[1]
            scorer = first_frame.index.levels[0]

            print('Scorer: ', scorer)
            print('Bodyparts ', bodyparts)

            for bpname in bodyparts:
                xypose = pose[scorer[0], bpname].drop(columns='likelihood')
                allbp[bpname][cam] = xypose.values
                if xypose.values is None:
                    print(key)
                    raise ValueError('Did not get pose value')

        # Insert stuff into MAIN CLASS
        self.insert1(key)

        # Update KEY with the pose datavand insert into correct PART subclass
        for bodypart in allbp.keys():
            for cam in cameras:
                if cam != 'overview': 
                    key[cam] = np.array([0]) 
                    continue

                classname = [s.capitalize() for s in bodypart.split('_')]
                classname = ''.join(classname)
                part = self.getattr(classname)

                cam_pose_data = allbp[bodypart][cam]
                if cam_pose_data is None:
                    Warning('No pose data detected', cam_pose_data, bodypart, cam)
                    key[cam] = np.array([0]) 
                else:
                    key[cam] = allbp[bodypart][cam]

            try:
                part.insert1(key)
            except:
                print('\n\nkey', key, '\n\n')
                print(self)
                part.insert1(key)



if __name__ == "__main__":
    import sys
    sys.path.append('./')
    from dj_config import start_connection
    start_connection()




