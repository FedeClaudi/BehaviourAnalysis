import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

from database.NewTablesDefinitions import *
from database.dj_config import start_connection


class analyse_all_trips:
    """ 
        get all trips data from the database
        divide them based on arm of orgin and return and trial or not
        plot shit

    """

    def __init__(self):
        self.tracking = pd.DataFrame((TrackingData.BodySegmentData & 'bpname=body'))





