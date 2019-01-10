import sys
sys.path.append('./')

from database.dj_config import start_connection
import datajoint as dj
dbname, _ = start_connection()
schema = dj.schema(dbname, locals())

@schema
class ConvertedTdms(dj.Manual):
    definition = """
        # Stores each .tdms video file if it was converted or not
        filename: varchar(128)                  # Name of the .tdms video
        ---
        converted: enum('Y', 'N')                 # single housed or group caged
        joined: enum('Y', 'N')                 # presence of wheel or other stuff in the cage
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