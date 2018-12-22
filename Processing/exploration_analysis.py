import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from collections import namedtuple

from database.NewTablesDefinitions import *
from database.dj_config import start_connection

from Utilities.file_io.files_load_save import load_yaml

class analyse_all_trips:
    """ 
        get all trips data from the database
        divide them based on arm of orgin and return and trial or not
        plot shit

    """

    def __init__(self):
        # Get tracking data
        all_bp_tracking = pd.DataFrame(TrackingData.BodyPartData.fetch())
        self.tracking = all_bp_tracking.loc[all_bp_tracking['bpname'] == 'body']

        # Get ROIs coordinates
        self.rois = self.get_rois()

        self.get_trips()

    def get_rois(self):
        roi = namedtuple('roi', 'x0 x1 width y0 y1 height')
        formatted_rois = {}
        rois_dict = load_yaml('Utilities\\video_and_plotting\\template_points.yml')

        for roi_name, cc in rois_dict.items():
            formatted_rois[roi_name] = roi(cc['x'], cc['x']+cc['width'], cc['width'],
                                            cc['y'], cc['y']+cc['height'], cc['height'])
        return formatted_rois

    def get_trips(self):
        def test_plot_rois_on_trace(x, y, rois):
            f, ax = plt.subplots()
            ax.plot(x, y, 'k')
            for roi, cc in rois.items():
                rect = patches.Rectangle((cc.x0,1000-cc.y0), cc.width,-cc.height,linewidth=1,edgecolor='r',facecolor='none', label=roi)
                ax.add_patch(rect)  
            ax.legend()

        trips=[]
        for idx, row in self.tracking.iterrows():
            tr = row['tracking_data']
            test_plot_rois_on_trace(tr[:,0], tr[:, 1], self.rois)

            f,ax = plt.subplots()
            when_in_rois = {}
            for roi, cc in self.rois.items():
                in_x =np.where((cc.x0<=tr[:, 0]) & (tr[:, 0]<= cc.x1))
                in_y = np.where((cc.y0<=tr[:, 1]) & (tr[:, 1]<= cc.y1))
                when_in_rois[roi] = [p for p in in_x[0] if p in in_y[0]]
                
                print('Spent {}/{} frames in {}'.format(len(when_in_rois[roi]), tr.shape[0], roi))

                xx = np.zeros(tr.shape[0])
                xx[when_in_rois[roi]] = 1

                ax.plot(np.diff(xx), label=roi)
            ax.legend()
            plt.show()


if __name__ == '__main__':
    analyse_all_trips()

