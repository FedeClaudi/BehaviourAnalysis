
import sys
sys.path.append('./')

from Processing.tracking_stats.math_utils import *
from Processing.tracking_stats.extract_velocities_from_tracking import complete_bp_with_velocity, get_body_segment_stats
from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame
from Utilities.file_io.files_load_save import load_yaml

import warnings
import cv2
import numpy as np
from collections import namedtuple
import os
import pandas as pd
from nptdms import TdmsFile
from database.dj_config import start_connection
import datajoint as dj
from database.TablesPopulateFuncs import *

dbname = start_connection()
schema = dj.schema(dbname, locals())


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
class CommonCoordinateMatrices(dj.Computed):
    definition = """
        # Stores matrixes used to align video and tracking data to standard maze model
        -> Sessions
        ---
        maze_model: longblob   # 2d array with image used for correction
        correction_matrix: longblob  # 2x3 Matrix used for correction
        alignment_points: longblob     # array of X,Y coords of points used for affine transform
    """

    def make(self, key):
        new_key = make_commoncoordinatematrices_table(key)
        self.insert1(new_key)

@schema
class Templates(dj.Imported):
    definition = """
    # stores the position of each maze template for one experiment
    -> Sessions
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
    b15: longblob
    """

    def _make_tuple(self, key):
        new_key = make_templates_table(key)
        self.insert1(new_key)

@schema
class Recordings(dj.Imported):
    definition = """
    # Within one session one may perform several recordings. Each recording has its own video and metadata files
    recording_uid: varchar(128)   # uniquely identifying name for each recording YYMMDD_MOUSEID_RECNUM
    -> Sessions
    ---
    """

    @schema
    class VideoFiles(dj.Part):
        definition = """
            # stores paths to video files
            -> Recordings
            camera_name: enum('overview', 'threat', 'catwalk', 'top_mirror', 'side_mirror')       # name of the camera
            ---
            filepath: varchar(256)          # path to the videofile
            """

    @schema
    class ConvertedVideoFiles(dj.Part):
        definition = """
            # stores paths to converted videos (from tdms to mp4), if no conversion is made then same as VideoFiles
            -> VideoFiles
            ---
            filepath: varchar(256)          # path to the new videofile
            """

    @schema
    class VideoMetadata(dj.Part):
        definition = """
            # It stores info about each video: frame size, fps...
            -> Recordings.ConvertedVideoFiles
            ---
            fps: int
            tot_frames: int
            frame_width: int
            frame_height: int
            frame_size: int         # number of bytes for the whole frame
            camera_offset_x: int            # camera offset
            camera_offset_y: int            # camera offset
        """

    @schema
    class PoseFiles(dj.Part):
        definition = """
            # stores paths to converted videos (from tdms to mp4), if no conversion is made then same as VideoFiles
            -> Recordings.ConvertedVideoFiles
            ---
            filepath: varchar(256)      # path to the .h5 file extracted with DLC
            """


    @schema
    class MetadataFiles(dj.Part):
        definition = """
            # stores paths to metadta .tdms files with video recording metadata
            -> Recordings.ConvertedVideoFiles
            ---
            filepath: varchar(256)      # path to the .tdms file 
            """

    @schema
    class AnalogInputFiles(dj.Part):
        definition = """
            # stores paths to AI .tdms files with analog input data
            -> Recordings
            ---
            filepath: varchar(256)      # path to the .tdms file 
            """


@schema
class AnalogInputs(dj.Imported):
    definition = """
        # Stores data from relevant AI channels recorded with NI board
        -> Recordings
        ---
        overview_camera_triggers: longblob      # Frame triggers signals efferent copy
        threat_camera_triggers: longblob
        speaker_signal: longblob                # HIGH when auditory stimulus being produced
        stimuli: longblob                       # timestamp and stimulus protocol
        ldr: longblob                           # light dependant resistor signal
    """


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


@schema
class TrackingData(dj.Computed):
    # Pose data for individual bp store X Y and Velocity
    # Pose data for body segments store Length,Angle, and Ang vel

    definition = """
    # Stores the DLC recording data for one Recording entry
    -> Recordings
    """

    def define_attrs(self):
        self.attributes = dict(
            LeftEar=self.LeftEar,
            LeftEye=self.LeftEye,
            Snout=self.Snout,
            RightEye=self.RightEye,
            RightEar=self.RightEar,
            Neck=self.Neck,
            RightShoulder=self.RightShoulder,
            RightHip=self.RightHip,
            TailBase=self.TailBase,
            Tail2=self.Tail2,
            Tail3=self.Tail3,
            LeftHip=self.LeftHip,
            LeftShoulder=self.LeftShoulder,
            Body=self.Body,
            HeadSegment=self.HeadSegment,
            EarsSegment=self.EarsSegment,
            UpperBodySegment=self.UpperBodySegment,
            LowerBodySegment=self.LowerBodySegment,
            TailSegment=self.TailSegment,
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

    def get_body_segments(self):
        segments = dict(
            Head=(self.attributes['Snout'], self.attributes['Neck']),
            Ears=(self.attributes['LeftEar'], self.attributes['RightEar']),
            UpperBody=(self.attributes['Neck'], self.attributes['Body']),
            LowerBody=(self.attributes['Body'], self.attributes['TailBase']),
            Tail=(self.attributes['TailBase'], self.attributes['Tail2']),
        )
        return segments

    # Docs on parts table
    # https://docs.datajoint.io/computation/Part-tables.html

    class LeftEar(dj.Part):
        definition = """
            -> TrackingData
            ---
            overview: longblob     # pose data extracted from overview camera - X,Y,VELOCITY
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

    class PositionOnMaze(dj.Part):
        """
            Pinpoints the position of the mouse on the maze by identifying which maze
            component (localised using templates) is closest to the mouse body
        """
        definition = """
            # Pinpoints the position of the mouse on the maze
            -> TrackingData
            ---
            maze_component: blob   # Name of the maze component closest to the mouse at each frame
            maze_position: blob    # position in px distance from shelter 
        """

    class HeadSegment(dj.Part):
        definition = """
            # From the snout to the neck
            -> TrackingData
            ---
            bp1: varchar(128)  # name of bodyparts
            bp2: varchar(128)
            length: longblob       # length in pixels
            theta: longblob        # clockwise angle relative to frame
            angvel: longblob       # angular velocity
        """

    class EarsSegment(dj.Part):
        definition = """
            # From the left ear to the right ear
            -> TrackingData
            ---
            bp1: varchar(128)  # name of bodyparts
            bp2: varchar(128)
            length: longblob       # length in pixels
            theta: longblob        # clockwise angle relative to frame
            angvel: longblob       # angular velocity
        """

    class UpperBodySegment(dj.Part):
        definition = """
            # From the neck to the body
            -> TrackingData
            ---
            bp1: varchar(128)  # name of bodyparts
            bp2: varchar(128)
            length: longblob       # length in pixels
            theta: longblob        # clockwise angle relative to frame
            angvel: longblob       # angular velocity
        """

    class LowerBodySegment(dj.Part):
        definition = """
            # From the body to tail base
            -> TrackingData
            ---
            bp1: varchar(128)  # name of bodyparts
            bp2: varchar(128)
            length: longblob       # length in pixels
            theta: longblob        # clockwise angle relative to frame
            angvel: longblob       # angular velocity
        """

    class TailSegment(dj.Part):
        definition = """
            # From the tail base to the tail2
            -> TrackingData
            ---
            bp1: varchar(128)  # name of bodyparts
            bp2: varchar(128)
            length: longblob       # length in pixels
            theta: longblob        # clockwise angle relative to frame
            angvel: longblob       # angular velocity
        """

    def fill_bp_data(self, pose_files, key):
        """fill_bp_data [fills in BP part classes with pose data]
        
        Arguments:
            pose_files {[list]} -- [list of .h5 pose files for a recording]
        """
        print('     ... filling in bodyparts PARTs classes')
        cameras = ['overview', 'threat', 'top_mirror', 'side_mirror']
        allbp = None
        # Create a dictionary with the data for each bp and each camera
        for cam in cameras:
            pfile = pose_files[cam]
            if pfile == 'nan':
                continue

            try:
                pose = pd.read_hdf(pfile)  # ? load pose data
            except FileNotFoundError:
                print('Could not open file: ', pfile)
                print(pose_files)
                raise FileExistsError()

            # Get the scorer name and the name of the bodyparts
            first_frame = pose.iloc[0]
            bodyparts = first_frame.index.levels[1]
            scorer = first_frame.index.levels[0]
            print('Scorer: ', scorer)
            print('Bodyparts ', bodyparts)

            if allbp is None:  # at the first iteration
                # initialise empty dict of dict
                allbp = {}
                for bp in bodyparts:
                    allbp[bp] = {cam: None for cam in cameras}

            # Get pose with X,Y coordinates
            for bpname in bodyparts:
                xypose = pose[scorer[0], bpname].drop(columns='likelihood')
                allbp[bpname][cam] = xypose.values

            # Insert entry into MAIN CLASS for this recording
            self.insert1(key)

            # Update KEY with the pose data and insert into correct PART subclass
            # ? first insert data into KEY
            for bodypart in allbp.keys():
                for cam in cameras:
                    if cam != 'overview':
                        key[cam] = np.array([0])
                        continue

                    # Get the PART subclass using the bodyparts name
                    classname = [s.capitalize() for s in bodypart.split('_')]
                    classname = ''.join(classname)
                    part = self.getattr(classname)

                    # Retrieve the data from the dictionary and insert
                    cam_pose_data = allbp[bodypart][cam]
                    if cam_pose_data is None:
                        Warning('No pose data detected',
                                cam_pose_data, bodypart, cam)
                        key[cam] = np.array([0])
                    else:
                        # Correct the tracking data to make it fit the standard maze model
                        uncorrected_bpdata = allbp[bodypart][cam]
                        ccm_entry = [
                            c for c in CommonCoordinateMatrices if c['uid'] == key['uid']]
                        if not ccm_entry:
                            raise FileNotFoundError(
                                'Couldnt find matching entry in CommonCoordinateMatrixes for recording: ', key['recording_uid'])
                        else:
                            M = ccm_entry[0]['correction_matrix']

                        corrected_bpdata = correct_tracking_data(
                            uncorrected_bpdata, M)
                        key[cam] = corrected_bpdata

                # ? Insert happens here
                try:
                    part.insert1(key)
                except:
                    # * Check what went wrong (print and reproduce error)
                    print('\n\nkey', key, '\n\n')
                    print(self)
                    part.insert1(key)

            # Get the position of the mouse on the maze and insert into correct part table
            body_tracking = allbp['body']['overview']
            if body_tracking.shape[1] > 2:
                body_tracking = body_tracking[:, :2]

            templates_idx = [i for i, t in enumerate(
                Templates.fetch()) if t['recording_uid'] == key['recording_uid']][0]
            rois = pd.DataFrame(Templates.fetch()).iloc[templates_idx]
            del rois['uid'], rois['session_name'], rois['recording_uid']

            shelter_roi_pos = rois['s']

            roi_at_each_frame = get_roi_at_each_frame(
                body_tracking, dict(rois))  # ? roi name
            position_at_each_frame = [(rois[r][0]-shelter_roi_pos[0],
                                       rois[r][1]-shelter_roi_pos[1])
                                      for r in roi_at_each_frame]  # ? distance from shelter
            warnings.warn(
                'Currently DJ canot store string of lists so roi_at_each_Frame is not saved in the databse')
            data_to_input = dict(
                recording_uid=key['recording_uid'],
                uid=key['uid'],
                session_name=key['session_name'],
                maze_component=-1,
                maze_position=position_at_each_frame,
            )
            self.PositionOnMaze.insert1(data_to_input)

    def fill_segments_data(self, key):
        print('     ... filling in segments PARTs classes')

        segments = self.get_body_segments()

        for name, segment in segments.items():
            print('             ... ', name)
            segment_data = key.copy()
            del segment_data['overview'], segment_data['threat'], segment_data['top_mirror'], segment_data['side_mirror']
            segment_data['bp1'] = segment[0].__name__
            segment_data['bp2'] = segment[1].__name__

            positions = []
            for bp in segment:
                xy = [p['overview']
                      for p in bp if p['recording_uid'] == key['recording_uid']][0]
                if xy.shape[1] > 2:
                    xy = xy[:, :2]
                positions.append(xy.T)

            segment_data['length'] = calc_distance_between_points_two_vectors_2d(
                positions[0].T, positions[1].T)
            segment_data['theta'] = calc_angle_between_vectors_of_points_2d(
                positions[0], positions[1])
            segment_data['angvel'] = calc_ang_velocity(segment_data['theta'])

            # Get PART subclass to add the segments to
            part = self.getattr(name+'Segment')

            # Insert the data in the table
            try:
                part.insert1(segment_data)
            except:
                raise ValueError('Faile to insert: ', segment_data,
                                 '\nWith Keys: ', segment_data.keys(), '\nInto: ', part.heading)

    def make(self, key):
        print('\n\nPopulating Tracking data\n', key)
        rec_name = key['recording_uid']
        pose_files = [ff for ff in Recordings.PoseFiles.fetch(
        ) if ff['recording_uid'] == rec_name][0]

        # Fill in the MAIN table and BodyParts PART tables
        self.fill_bp_data(pose_files, key)

        # FIll in the SEGMENTS PART tables
        self.fill_segments_data(key)

if __name__ == "__main__":
    import sys
    sys.path.append('./')
    from dj_config import start_connection
    start_connection()
