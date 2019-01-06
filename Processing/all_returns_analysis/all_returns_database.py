import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from pandas.plotting import scatter_matrix
from collections import namedtuple
from itertools import combinations

from database.NewTablesDefinitions import *
from database.dj_config import start_connection

from Processing.tracking_stats.math_utils import line_smoother
from Utilities.file_io.files_load_save import load_yaml
from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame


class analyse_all_trips:
    """ 
        get all trips data from the database
        divide them based on arm of orgin and return and trial or not
    """

    def __init__(self, erase_table=False, fill_in_table=False):
        if erase_table:
            AllTrips.drop()
            print('Table erased, exiting...')
            sys.exit()

        if fill_in_table:
            # Get tracking data
            print('     ... fetching data')
            #  (mouse & 'dob > "2017-01-01"' & 'sex = "M"').fetch()
            # all_bp_tracking = pd.DataFrame(TrackingData.BodyPartData.fetch())
            fetched = (TrackingData.BodyPartData & 'bpname = "body"').fetch()

            all_bp_tracking = pd.DataFrame(fetched)
            self.tracking = all_bp_tracking.loc[all_bp_tracking['bpname'] == 'body']

            # bodyaxis_tracking = pd.DataFrame(TrackingData.BodySegmentData.fetch())
            self.ba_tracking = pd.DataFrame((TrackingData.BodySegmentData & 'bp1 = "body_axis"').fetch())
            self.stimuli = pd.DataFrame(BehaviourStimuli.fetch())

            print('... ready')

            # Get ROIs coordinates
            self.rois = self.get_rois()

            # Get all good trips
            self.all_trips = []
            self.trip = namedtuple('trip', 'shelter_exit threat_enter threat_exit shelter_enter time_in_shelter tracking_data is_trial recording_uid')
            
            print('     ... getting trips')
            self.get_trips()

            # Insert good trips into trips table
            self.table = AllTrips()
            self.insert_trips_in_table()

            print(self.table)

    #####################################################################
    #####################################################################

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
        trips=[]
        goodtime = namedtuple('gt', 'shelter_exit threat_enter threat_exit shelter_enter time_in_shelter')
        for idx, row in self.tracking.iterrows():
            tr = row['tracking_data']
            print(row['recording_uid'], idx, ' of ', self.tracking.shape)
            in_rois = {}
            in_roi = namedtuple('inroi', 'ins outs')
            print('  ... getting times in rois')
            for i, (roi, cc) in enumerate(self.rois.items()):
                # Get time points in which mouse is in roi
                x = line_smoother(tr[:, 0])
                y = line_smoother(tr[:, 1])
                in_x =np.where((cc.x0<=x) & (x<= cc.x1))
                in_y = np.where((cc.y1<=y) & (y<= cc.y0))

                in_xy = np.intersect1d(in_x, in_y)
                in_xy = in_xy[in_xy>=9000]  # ? not sure what this line does
                raise NotImplementedError('Check what the line above does you dummy')

                # get times at which it enters and exits
                xx = np.zeros(tr.shape[0])
                xx[in_xy] = 1
                enter_exit = np.diff(xx)
                enters, exits = np.where(np.diff(xx)>0)[0], np.where(np.diff(xx)<0)[0]
                in_rois[roi] = in_roi(list(enters), list(exits))

            # Get bodylength and add it to tracking data
            try:  # can only do this if we have body length in tracking data
                blen = self.ba_tracking.loc['recording_uid' == row['recording_uid']]['tracking_data'].values
            except:
                warnings.warn('No body length recorded')
                blen = np.zeros((tr.shape[0],1))
            tr = np.append(tr, blen, axis=1)

            # get complete s-t-s trips
            print(' ... getting good trips')
            good_times = []
            for sexit in in_rois['shelter'].outs:
                # get time in which it returns to the shelter 
                try:
                    next_in = [i for i in in_rois['shelter'].ins if i > sexit][0]
                    time_in_shelter = [i for i in in_rois['shelter'].outs if i > next_in][0]-next_in  # time spent in shelter after return
                except:
                    break

                # Check if it reached the threat
                at_threat = [i for i in in_rois['threat'].ins if i > sexit and i < next_in]              
                if at_threat:
                    tenter = at_threat[0]
                    try:
                        texit = [t for t in in_rois['threat'].outs if t>tenter and t<next_in][-1]
                    except:
                        pass  # didn't reach threat, don't append the good times
                    else:
                        gt = goodtime(sexit, tenter, texit, next_in, time_in_shelter)
                        good_times.append(gt)

            # Check if trip includes trial and add to al trips dictionary
            print(' ... checking if trials')
            for g in good_times:
                rec_stimuli = self.stimuli.loc[self.stimuli['recording_uid'] == row['recording_uid']]
                stims_in_time = [s for s in rec_stimuli['stim_start'].values if g[0]<s<g[2]]
                if stims_in_time:
                    has_stim = 'true'
                else:
                    has_stim = 'false'

                # 'shelter_exit threat_enter threat_exit shelter_enter time_in_shelter tracking_data is_trial recording_uid'
                'shelter_exit threat_enter shelter_enter time_in_shelter'
                endtime = g.shelter_enter+gtime_in_shelter
                self.all_trips.append(self.trip(g.shelter_exit, g.threat_enter, g.threat_exit, g.shelter_enter,
                            g.time_in_shelter, tr[g.shelter_exit:endtime, :], has_stim, row['recording_uid']))

    def insert_trips_in_table(self):
        for i, trip in enumerate(self.all_trips): 
            key = trip._asdict()
            key['trip_id'] = i
            try:
                self.table.insert1(key)
                print('inserted: ', key['recording_uid'])
            except:
                print(' !!! - did not insert !!! - ', key['recording_uid'])
          

if __name__ == '__main__':
    print('Ready')
    analyse_all_trips(erase_table=False, fill_in_table=True)



