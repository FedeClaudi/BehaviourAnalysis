import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.image as mpimg
import pandas as pd
import os

from database.NewTablesDefinitions import *
from database.database_fetch import *

from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame, get_arm_given_rois, convert_roi_id_to_tag
from Processing.tracking_stats.math_utils import *
from Utilities.video_and_plotting.video_editing import Editor





class plotPR:
    def __init__(self):
        self.save_fld = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\plots\\p(R)'


    def plot_pr_by_exp(self):
        experiments = set(AllTrials.fetch('experiment_name'))

        por_origins, por_escapes = [], []
        for exp in sorted(experiments):        
            origins, escapes = (AllTrials &  "experiment_name='{}'".format(exp) & "is_escape='{}'".format('true')).fetch("origin_arm", "escape_arm")
        
            por_origins.append(-list(origins).count('Right_Medium')/len(origins))
            por_escapes.append(list(escapes).count('Right_Medium')/len(escapes))

        f, ax = plt.subplots()
        x = np.arange(len(por_origins))
        ax.bar(x, por_origins)
        ax.bar(x, por_escapes)

        ax.set(xticks=[])
        ax.xlables = experiments

        

if __name__ == "__main__":
    ppr = plotPR()
    ppr.plot_pr_by_exp()

    plt.show()