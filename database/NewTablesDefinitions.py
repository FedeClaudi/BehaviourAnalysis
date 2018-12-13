
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
            threat_camera_triggers: longblob            # a
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
            fps: int                # fps
            tot_frames: int
            frame_width: int
            frame_height: int
            frame_size: int         # number of bytes for the whole frame
            camera_offset_x: int            # camera offset
            camera_offset_y: int            # camera offset
        """


    def make(self, key):
        make_videofiles_table(self, key, Recordings, VideosIncomplete)


@schema
class VideosIncomplete(dj.Imported):
    definition = """
        # Stores the ID of Videos that have missing files or items and what is missing
        -> VideoFiles
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
    stim_metadata: longblob         # list of other stuff ? 
    video: varchar(256)             # name of corresponding video
    """

    def make(self, key):
        make_behaviourstimuli_table(self, key, Recordings)


if __name__ == "__main__":
    import sys
    sys.path.append('./')
    from dj_config import start_connection
    start_connection()

    # VideoFiles.drop_quick()
    print(VideoFiles.Metadata.heading)
