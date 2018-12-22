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

    def __init__(self, fill_in_table=True):
        if fill_in_table:
            # Get tracking data
            all_bp_tracking = pd.DataFrame(TrackingData.BodyPartData.fetch())
            self.tracking = all_bp_tracking.loc[all_bp_tracking['bpname'] == 'body']
            self.stimuli = pd.DataFrame(BehaviourStimuli.fetch())

            # Get ROIs coordinates
            self.rois = self.get_rois()

            # Get all good trips
            self.all_trips = []
            self.trip = namedtuple('trip', 'shelter_exit shelter_enter tracking_data is_trial recording_uid')
            self.get_trips()

            # Insert good trips into trips table
            self.table = AllTrips()
            self.insert_trips_in_table()

    def get_rois(self):
        roi = namedtuple('roi', 'x0 x1 width y0 y1 height')
        formatted_rois = {}
        rois_dict = load_yaml('Utilities\\video_and_plotting\\template_points.yml')

        for roi_name, cc in rois_dict.items():
            if roi_name == 'translators': continue
            formatted_rois[roi_name] = roi(cc['x'], cc['x']+cc['width'], cc['width'],
                                            cc['y'], cc['y']-cc['height'], cc['height'])
        return formatted_rois

    def get_trips(self):
        def test_plot_rois_on_trace(x, y, rois, inx, iny, inboth):
            f, ax = plt.subplots()
            ax.plot(x, y, 'k')
            for roi, cc in rois.items():
                rect = patches.Rectangle((cc.x0,cc.y0), cc.width,-cc.height,linewidth=1,edgecolor='r',facecolor='none', label=roi)
                ax.add_patch(rect)  
            ax.legend()

            f, ax = plt.subplots()
            ax.plot(x, y, 'k', alpha=.5)
            # ax.plot(x[inx], y[inx], 'r',alpha=.75, label='x')
            # ax.plot(x[iny], y[iny], 'b',alpha=.75, label='y')
            ax.plot(x[inboth], y[inboth], 'r')
            # ax.legend()
            plt.show()

        trips=[]
        for idx, row in self.tracking.iterrows():
            if idx < 4 : continue
            tr = row['tracking_data']

            f,ax = plt.subplots()
            in_rois = {}
            in_roi = namedtuple('inroi', 'ins outs')

            all_signal = np.zeros((tr.shape[0]))
            for i, (roi, cc) in enumerate(self.rois.items()):
                # Get time points i which mouse is in roi
                in_x =np.where((cc.x0<=tr[:, 0]) & (tr[:, 0]<= cc.x1))
                in_y = np.where((cc.y1<=tr[:, 1]) & (tr[:, 1]<= cc.y0))
                in_xy = [p for p in in_x[0] if p in in_y[0]]
                
                # get times at which it enters and exits
                xx = np.zeros(tr.shape[0])
                xx[in_xy] = 1
                all_signal[in_xy] = 1
                enter_exit = np.diff(xx)
                enters, exits = np.where(np.diff(xx)>0)[0], np.where(np.diff(xx)<0)[0]
                in_rois[roi] = in_roi(list(enters), list(exits))
                print('Roi ', roi, len(enters), ' enters and ', len(exits), ' exits')
                
                # test_plot_rois_on_trace(tr[:,0], tr[:, 1], self.rois, in_x, in_y,  when_in_rois[roi])
                ax.plot(np.add(np.diff(xx), i*3), label=roi)
            all_signal = np.diff(all_signal)


            # get complete s-t-s trips
            good_times = []
            for sexit in in_rois['shelter'].outs:
                # get time in which it returns to the shelter 
                try:
                    next_in = [i for i in in_rois['shelter'].ins if i > sexit][0]
                except:
                    break

                # Check if it reached the threat
                at_threat = [i for i in in_rois['threat'].ins
                            if i > sexit and i < next_in]
                if at_threat:
                    good_times.append((sexit, next_in))

            # Check if trip includes trial and add to al trips dictionary
            for g in good_times:
                rec_stimuli = self.stimuli.loc[self.stimuli[recording_uid] == row['recording_uid']]
                stims_in_time = [s for s in rec_stimuli['stim_start'].values if g[0]<s<g[1]]
                if stims_in_time:
                    has_stim = True
                else:
                    has_stim = False
                self.all_trips.append(
                    self.trip(g[0], g[1], tr[g[0]:g[1], :]), has_stim, row['recording_uid'])

            # Plot for debugging
            # f2, ax2 = plt.subplots()
            # for good in good_times:
            #     ax.plot([good[0], good[1]], [-2, -2], 'g')    
            #     # ax2.plot(tr[good[0]:good[1],0], tr[good[0]:good[1],1])
            #     ax2.hist(durs)

            # ax.plot(np.add(all_signal, -5), 'k')
            # ax.legend()
            # plt.show()

    def insert_trips_in_table(self):
        for i, trip in enumerate(self.trips): 
            key = dict(trip)
            key['trip_id'] = i
            self.table.insert1(key)


if __name__ == '__main__':
    analyse_all_trips()

