import sys
sys.path.append('./')

from database.NewTablesDefinitions import *
import datajoint as dj
from database.dj_config import start_connection

dbname, _ = start_connection()
schema = dj.schema(dbname, locals())

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
    print(VisualStimuliMetadata())