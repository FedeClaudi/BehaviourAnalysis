
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
    -> Experiments      # ame of the experiment
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
        top_pad: int            # y-shift
        side_pad: int            # x-shift
    """

    def make(self, key):
        make_commoncoordinatematrices_table(self, key, Sessions, VideoFiles)

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

    def make(self, key):
        new_key = make_templates_table(key, CommonCoordinateMatrices)
        self.insert1(new_key)

@schema
class Recordings(dj.Imported):
    definition = """
        # Within one session one may perform several recordings. Each recording has its own video and metadata files
        recording_uid: varchar(128)   # uniquely identifying name for each recording YYMMDD_MOUSEID_RECNUM
        -> Sessions
        ---
        software: enum('behaviour', 'mantis')
        ai_file_path: varchar(256)    # path to mantis .tdms file with analog inputs and stims infos
    """
    class AnalogInputs(dj.Part):
        definition = """
            # Stores data from relevant AI channels recorded with NI board
            -> Recordings
            ---
            tstart: float                           # t0 from mantis manuals .tdms
            overview_camera_triggers: longblob      # Frame triggers signals efferent copy
            threat_camera_triggers: longblob        # a
            audio_irled: longblob                   # HIGH when auditory stimulus being produced
            audio_signal: longblob                  # voltage from amplifier to speaker
            manuals_names: longblob                 # list of strings of name of manual protocols
            manuals_timestamps: longblob            # list of floats of timestamps of manual protocols
            ldr: longblob                           # light dependant resistor signal
        """

    def print(self):
        print(self.AnalogInputs.heading)
        for line in self.AnalogInputs.fetch():
            print('tule', line)


    def make(self, key):
        make_recording_table(self, key)

@schema
class VideoFiles(dj.Imported):
    definition = """
        # stores paths to video files and all metadata and posedata
        -> Recordings
        camera_name: enum('overview', 'threat', 'catwalk', 'top_mirror', 'side_mirror')       # name of the camera
        ---
        video_filepath: varchar(256)          # path to the videofile
        converted_filepath: varchar(256)      # path to converted .mp4 video, if any, else is same as video filepath    
        metadata_filepath: varchar(256)       # if acquired with mantis a .tdms metadata file was produced, path ot it.
        pose_filepath: varchar(256)           # path to .h5 pose file
        """

    class Metadata(dj.Part):
        definition = """
            # contains info about each video
            -> VideoFiles
            ---
            fps: int                        # fps
            tot_frames: int
            frame_width: int
            frame_height: int
            frame_size: int                 # number of bytes for the whole frame
            camera_offset_x: int            # camera offset
            camera_offset_y: int            # camera offset
        """

    def make(self, key):
        make_videofiles_table(self, key, Recordings, VideosIncomplete)

@schema
class VideosIncomplete(dj.Manual):
    definition = """
        # Stores the ID of Videos that have missing files or items and what is missing
        -> Recordings
        camera_name: enum('overview', 'threat', 'catwalk', 'top_mirror', 'side_mirror')       # name of the camera
        ---
        conversion_needed: enum('true', 'false')
        dlc_needed: enum('true', 'false')
    """

@schema
class BehaviourStimuli(dj.Computed):
    definition = """
    # Stimuli of sessions recorded with old behaviour software
    -> Recordings
    stimulus_uid: varchar(128)  # uniquely identifying ID for each trial YYMMDD_MOUSEID_RECNUM_TRIALNUM
    ---
    stim_type: varchar(128)
    stim_start: int                 # number of frame at start of stim
    stim_duration: int              # duration in frames
    stim_name: varchar(128)         # list of other stuff ? 
    video: varchar(256)             # name of corresponding video
    """

    def make(self, key):
        make_behaviourstimuli_table(self, key, Recordings, VideoFiles)


@schema 
class MantisStimuli(dj.Computed):
    definition = """
        # stores metadata regarding stimuli with Mantis software
        -> Recordings
        stimulus_uid: varchar(128)      # uniquely identifying ID for each trial YYMMDD_MOUSEID_RECNUM_TRIALNUM
        ---
        overview_frame: int             # frame number in overview camera (of onset)
        threat_frame: int               # frame number in threat camera (of onset)
        overview_frame_off: int
        threat_frame_off: int
        duration: int                   # duration in seconds
        stim_type: varchar(128)         # audio vs visual
        stim_name: varchar(128)         # name 
    """

    def make(self, key):
        make_mantistimuli_table(self, key, Recordings, VideoFiles)    

@schema
class TrackingData(dj.Computed):
    definition = """
        # store dlc data for bodyparts and body segments
        -> VideoFiles
    """

    class BodyPartData(dj.Part):
        definition = """
            # stores X,Y,Velocity... for a single bodypart
            -> TrackingData
            bpname: varchar(128)        # name of the bodypart
            ---
            tracking_data: longblob     # pandas dataframe with X,Y,Velocity, MazeComponent ... 
        """

    class BodySegmentData(dj.Part):
        definition = """
            # stores length,orientation... for a body segment between two body parts
            -> TrackingData
            bp1: varchar(128)           # name of the first bodypart
            bp2: varchar(128)
            ---
            tracking_data: longblob     # pandas dataframe with Length, Orientation, Ang Velovity... 
        """

    def define_bodysegments(self):
        segment = namedtuple('seg', 'bp1 bp2')
        self.segments = dict(
            head = segment('snout', 'neck'),
            ears=segment('left_ear', 'right_ear'),
            body_upper=segment('neck', 'body'),
            body_lower=segment('body', 'tail_base'),
            
        )
        # tail1=segment('tail_base', 'tail_2'),
        # tail2=segment('tail_2', 'tail_3'),

    def make(self, key):
        self.define_bodysegments()
        make_trackingdata_table(self, key, VideoFiles, CommonCoordinateMatrices, Templates)


@schema
class BehaviourTrialOutcomes(dj.Manual):
    definition = """
        # For each stimulus stores info about the escape...
        stimulus_uid: varchar(128)          # name of the stimulus
        ---
        criteria: varchar(128)              # e.g. if a time limit is set to specify if a trial is escape or not
        threshold: int                      # the value associated to the criteria (e.g. num of seconds)
        escape: enum('true', 'false')
        origin_arm: varchar(128)
        escape_arm: varchar(128)
        x_y_theta: longblob                # x,y position and body orientation at stimulus onset
        reaction_time: int                 # reaction time, if calculated
        time_to_shelter: int               # number of seconds before the shelter is reached 
    """

@schema
class DLCmodels(dj.Lookup):
    definition = """
        # It got pointers to dlc models so that they can be used for analysing videos
        model_name: varchar(256)                        # name given to the model
        ---
        cfg_file_path: varchar(256)                     # path to the cfg.yml file
        camera: enum('overview', 'threat', 'mirrors')   # for which kind of video it can be used
    """

    def populate(self):
        make_dlcmodels_table(self)



if __name__ == "__main__":
    import sys
    sys.path.append('./')
    from dj_config import start_connection
    start_connection()

    # VideoFiles.drop_quick()
    print(VideoFiles.Metadata.heading)
