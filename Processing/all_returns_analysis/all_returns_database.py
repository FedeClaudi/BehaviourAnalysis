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
            self.trip = namedtuple('trip', 'shelter_exit threat_enter threat_exit shelter_enter time_in_shelter tracking_data is_trial recording_uid duration max_speed is_escape arm_taken')
            
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
                # raise NotImplementedError('Check what the line above does you dummy')

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

                # 'shelter_exit threat_enter threat_exit shelter_enter time_in_shelter tracking_data is_trial recording_uid duration max_speed is_escape arm_taken')

                # Get remaining variables
                endtime = g.shelter_enter+g.time_in_shelter
                tracking_data = tr[g.shelter_exit:endtime, :]

                duration = (g.shelter_enter - g.threat_exit)/30 # ! hardcoded fps
                smooth_speed = line_smoother(tracking_data[:,2])
                max_speed = np.max(smooth_speed)
                if duration <= 3: # ! hardcoded arbritary variable
                    is_escape = 'true'
                else:
                    is_escape = 'false'

                x_displacement = self.get_x_displacement(tracking_data[:, 0], tracking_data[:, 1], g.threat_exit, g.shelter_exit, g.shelter_enter)
                arms_lims = dict(Left_Far=(-10000, -251),
                                Left_Medium=(-250, -100),
                                Centre=(-99, 99),
                                Right_Medium= (100, 250),
                                Right_Far= (251, 10000))

                for k, (x0, x1) in arms_lims.items():
                    if x0 <= x_displacement <= x1:
                        arm_taken = k

                self.all_trips.append(self.trip(g.shelter_exit, g.threat_enter, g.threat_exit, g.shelter_enter,
                            g.time_in_shelter, tracking_data, has_stim, row['recording_uid'], duration, max_speed, is_escape, arm_taken))
                # For each good trip get duration, max_v....
                duration = ()

    @ staticmethod
    def get_x_displacement(x, y, threat_exit, shelter_exit, shelter_enter):
        """[Gets the max X displacement during escape, used to figure out which arm was taken ]
        
        Arguments:
            x {[np.array]} -- [description]
            y {[np.array]} -- [description]
            threat_exit {[int]} -- [description]
            shelter_exit {[int]} -- [description]
            shelter_enter {[unt]} -- [description]
        """

        # Get x between the time T is left and the mouse is at the shelter
        t0, t1 = threat_exit-shelter_exit, shelter_enter-shelter_exit
        x = line_smoother(np.add(x[t0:t1], -500)) # center on X
        y = line_smoother(y[t0:t1])

        # Get left-most and right-most points
        left_most, right_most = min(x), max(x)
        if abs(left_most)>=abs(right_most):
            x_displacement = left_most
        else:
            x_displacement = right_most

        # Plt for debug
        # idx = np.where(x == tokeep)[0]
        # f, ax = plt.subplots()
        # plt.plot(line_smoother(np.add(row['tracking_data'][:, 0], -500)),
        #         line_smoother(row['tracking_data'][:, 1]), 'k')
        # ax.plot(x, y, 'g', linewidth=3)
        # ax.plot(x[idx], y[idx], 'o', color='r')
        # plt.show()

        # if self.plot:
        #     sampl = np.random.uniform(low=0.0, high=10.0, size=(len(x_displacement),))
        #     f,ax = plt.subplots()
        #     ax.scatter(x_displacement, sampl, s=10, c='k', alpha=.5)
        #     ax.set(title='x displacement')

        return x_displacement

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
    # analyse_all_trips(erase_table=False, fill_in_table=True)

    print(AllTrips())

