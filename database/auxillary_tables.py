import sys
sys.path.append('./')

from database.dj_config import start_connection
import datajoint as dj
from database.TablesPopulateFuncs import *
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






