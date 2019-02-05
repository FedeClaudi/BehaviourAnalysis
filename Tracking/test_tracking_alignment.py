import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('./')
from database.NewTablesDefinitions import *

from Processing.rois_toolbox.rois_stats import load_rois

"""
    Plot tracking data from all sesssions in the database to check that they match the maze template
"""

def get_roi_center(points):
    x = points[1] + (points[3] / 2)
    y = points[0] + (points[2] / 2)
    return x, y



# Get tracking data
fetched = (TrackingData.BodyPartData & 'bpname = "body"').fetch()
all_bp_tracking = pd.DataFrame(fetched)
body_tracking = all_bp_tracking.loc[all_bp_tracking['bpname'] == 'body']

# Get maze model
maze_model = cv2.imread('Utilities\\video_and_plotting\\mazemodel.png')
maze_model = cv2.resize(maze_model, (1000, 1000))
maze_model = cv2.cvtColor(maze_model,cv2.COLOR_RGB2GRAY)

# Get ROIs
paths = load_yaml('paths.yml')
rois = load_yaml(paths['maze_model_templates'])


f, axarr = plt.subplots(nrows=2)

#  Show maze model
axarr[0].imshow(maze_model)
axarr[1].imshow(maze_model)

# plot ROIs
for name, roi in rois.items():
    if name in ['uid', 'session_name']: continue
    x, y  = get_roi_center(roi)
    axarr[0].plot(x, y, 'o', color='k')

# Plot tracking
# for i, row in body_tracking.iterrows():
#     try:
#         axarr[1].scatter(row['tracking_data'][6000:16000, 0], row['tracking_data'][6000:16000, 1], 
#                         c= row['tracking_data'][6000:16000, -1],
#                         alpha=.01, s=10)
#     except:
#         pass
    
    

shelterx, sheltery = get_roi_center(rois['b4'])
axarr[1].plot(shelterx, sheltery, 'o', color='r')
axarr[1].set(ylim=[1000, 0])
plt.show()




