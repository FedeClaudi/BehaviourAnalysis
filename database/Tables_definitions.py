import sys
sys.path.append('./')  

import datajoint as dj
from dj_config import start_connection
dbname = start_connection()
schema = dj.schema(dbname, locals())

import pandas as pd
import os

from Utilities.file_io import load_yaml

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
    ---
    -> Sessions
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

    def _make_tuple(self, session):
        """ Populate the Recordings table """
        
        paths = load_yaml('paths.yml')
        raw_video_folder = os.path.join(paths['raw_data_folder'], paths['raw_video_folder'])
        raw_metadata_folder = os.path.join(paths['raw_data_folder'], paths['raw_metadata_folder'])

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
                posefile = [os.path.join(self.tracked_data_folder, f) for f in os.listdir(self.tracked_data_folder)
                            if rec_name == os.path.splitext(f)[0].split('Deep')[0] and '.pickle' not in f]
                if not posefile:
                    print('didnt find pose file, trying harder')
                    posefile = [os.path.join(self.tracked_data_folder, f) for f in os.listdir(self.tracked_data_folder)
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
                videofiles['overview'] = os.path.join(self.raw_video_folder, vid)

                convertedvideofiles = videofiles.copy()

                metadatafiles = {c: 'nan' for c in cameras}
                metadatafiles['overview'] = os.path.join(self.raw_metadata_folder, met)

                posefiles = {c: 'nan' for c in cameras}
                posefiles['overview'] = posefile

                # actually insert
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
        nplatf, nbridges = 9, 14
        platforms = ['p'+str(i) for i in range(nplatf + 1)]
        bridges = ['b'+str(i) for i in range(nbridges + 1)]
        all_components = ['s', 't']
        all_components.extend(platforms)
        all_components.extend(bridges)

        # Get matched components for recording
        session = key['session_name']
        experiment = key['experiment_name']
        templates_fld = key['templates_folder']
        video = key['video_file_path']
        matched = get_rois_from_templates(session, video, templates_fld)

        # Prepare data to insert into the table
        data_to_input =  {(n if n in matched.keys() else n):(matched[n] if n in matched.keys() else 0)
                           for n in all_components}
        for k,v in data_to_input:
            key[k] = v

        # Insert
        self.insert1(key)

@schema
class Stimuli(dj.Computed):
    definition = """
    # Metadata of each trial (e.g. stim type and frame of onset)
    -> Recordings
    uid: varchar(128)  # uniquely identifuing ID for each trial YYMMDD_MOUSEID_RECNUM_TRIALNUM
    ---
    stim_type: varchar(128)
    stim_start: int   # number of frame at start of stim
    stim_duration: int   # duration in frames
    stim_metadata: longblob  # list of other stuff ? 
    """

    def _make_tuple(self, key):
        # TODO Mantis
        # TODO more metadata
        tdmspath = key['metadata_file_path']
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
        for i, k in enumerate(sorted(stimuli.keys())):
            stim = stimuli[k]

            data_to_input = dict(
                uid=recording_uid + '_{}'.format(i),
                stim_type = stim_type,
                stim_start = int(framen),
                stim_duration = stim_duration,
                stim_metadata = 0  # ! <- 
            )
            self.insert1(data_to_input)

@schema
class TrackingData(dj.Computed):
    definition = """
    # Stores the DLC recording data for one Recording entry
    -> Recordings
    """
    # Docs on parts table 
    # https://docs.datajoint.io/computation/Part-tables.html

    class LeftEar(dj.Part):
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


if __name__ == "__main__":
    import sys
    sys.path.append('./')
    from dj_config import start_connection
    start_connection()




