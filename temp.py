import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np  
import matplotlib.pyplot as plt
sys.path.append('./')   
from nptdms import TdmsFile
from database.dj_config import start_connection
from database.NewTablesDefinitions import *


def run():

    td = TrackingData()

    for tracking in td.fetch(as_dict=True):
        bodyparts = [bp for bp in td.BodyPartData.fetch(as_dict=True)
                    if bp['recording_uid']==tracking['recording_uid']]
        bodysegments = [bp for bp in td.BodySegmentData.fetch(as_dict=True)
                        if bp['recording_uid']==tracking['recording_uid']]  
    
        [print(bp['bpname']) for bp in bodyparts]
        [print(s['bp1'], s['bp2']) for s in bodysegments]

        f, ax = plt.subplots()
        ax.plot(bodyparts[0]['tracking_data'][10000:10020, 0],
                bodyparts[0]['tracking_data'][10000:10020, 1])
        ax.plot(bodyparts[2]['tracking_data'][10000:10020, 0], bodyparts[2]['tracking_data'][10000:10020, 1])
        ax.set(xlim=[0, 1000])
        plt.show()


if __name__ == "__main__":
        run()


