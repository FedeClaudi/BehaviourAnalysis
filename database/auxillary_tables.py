import sys
sys.path.append('./')

import datajoint as dj
from database.dj_config import start_connection
from database.NewTablesDefinitions import *


# TODO WIP
@schema
class FrameTimes(dj.Manual):
    """[Stores the time of each frame for the overview and threat videos so that they can later be aligned]
    """

    definition = """
        -> Recordings
        ---
        overview_frames_timestamps: longblob
        threat_frames_timestamps: longblob 
    """


# TODO WIP
@schema
class VisualStimuliMetadata(dj.Manual):

    """
        columns: ['Stim type', 'color', 'end_size', 'expand_time', 'modality', 'off_time', 'on_time', 'pos', 'repeats', 'start_size', 'stim_count', 'stim_name',
            'stim_start', 'type', 'units']
        vals: ['loom' 0.0 15.0 360.0 'linear' 250.0 240.0 '480, 75' 1.0 0.5 0.0 'FC_1ll_30_B_360ms.yml' '15:34' 'loom' 'degs']

          background_luminosity: 125
    """
    
    # TODO make it depend on mantis stimuli to get the correct dependency
    definition = """
        - stim_name:            varchar(128)    # just a number
        ---
        stim_type:              varchar(128)    # loom, grating...
        modality:               varchar(128)    # linear, exponential. 
        params_file:            varchar(128)    # name of the .yml file with the params
        time:                   varchar(128)    # time at which the stim was delivered
        units:                  varchar(128)    # are the params defined in degrees, cm ...

        start_size:             float       
        end_size:               float
        expansion_time:         float
        on_time:                float
        off_time:               float

        color:                  float
        backgroun_color:        float
        contrast:               float

        position:               blob
        repeats:                int
        sequence_number:        float           # sequential stim number in the session
    """


@schema
class VideoTdmsMetadata(dj.Manual):
    definition = """
        # Stores the metadata for the videos to be converted, to be used for conversion and to check that everything went fine
        videopath: varchar(256)
        ---
        width: int
        height: int
        number_of_frames: int
        fps: int
    """


if __name__ == "__main__":
    # VideoTdmsMetadata().drop()
    print(VideoTdmsMetadata())