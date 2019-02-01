import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('./')
from database.NewTablesDefinitions import *

"""
    get tracking data
"""

fetched = (TrackingData.BodyPartData & 'bpname = "body"').fetch()
all_bp_tracking = pd.DataFrame(fetched)
body_tracking = all_bp_tracking.loc[all_bp_tracking['bpname'] == 'body']


maze_model = cv2.imread('Utilities\\video_and_plotting\\mazemodel.png')
maze_model = cv2.resize(maze_model, (1000, 1000))
maze_model = cv2.cvtColor(maze_model,cv2.COLOR_RGB2GRAY)



f, ax = plt.subplots()

ax.imshow(maze_model)

# Plot the tracking data of each recording
for i, row in body_tracking.iterrows():
    try:
        ax.scatter(row['tracking_data'][6000:, 0], row['tracking_data'][6000:, 1], alpha=.01)
    except:
        pass



plt.show()




