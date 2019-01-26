import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from pandas.plotting import scatter_matrix
from collections import namedtuple
from itertools import combinations
import time

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

        self.naughty_experiments = ['PathInt', 'Lambda Maze', 'Square Maze']
        self.good_experiments = ['PathInt2']
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
            self.trip = namedtuple('trip', 'shelter_exit threat_enter threat_exit shelter_enter time_in_shelter tracking_data is_trial recording_uid duration max_speed is_escape arm_taken experiment_name')
            
            print('     ... getting trips')
            self.get_trips()

            # Insert good trips into trips table
            self.table = AllTrips()
            self.insert_trips_in_table()

            print(self.table)

    #####################################################################
    #####################################################################

    def get_rois(self):
        # roi = namedtuple('roi', 'x0 x1 width y0 y1 height')
        # formatted_rois = {}
        # rois_dict = load_yaml('Utilities\\video_and_plotting\\template_points.yml')

        # for roi_name, cc in rois_dict.items():
        #     if roi_name == 'translators': continue
        #     formatted_rois[roi_name] = roi(cc['x'], cc['x']+cc['width'], cc['width'],
        #                                     cc['y'], cc['y']-cc['height'], cc['height'])
        # return formatted_rois
        rois = dict(shelter=0, threat=1)
        return rois

    def get_trips(self):
        def checl_tracking_plotter(tracking_data):
            f, ax = plt.subplots()
            ax.scatter(tracking_data[:, 0], tracking_data[:, 1], c=tracking_data[:, -1], s=10)
            plt.show()


        recordings = pd.DataFrame(Recordings().fetch())
        sessions = pd.DataFrame(Sessions().fetch())
        trips=[]
        goodtime = namedtuple('gt', 'shelter_exit threat_enter threat_exit shelter_enter time_in_shelter experiment_name')
        for idx, row in self.tracking.iterrows():
            # To exclude trials from unwanted experiment...
            rec = recordings.loc[recordings['recording_uid'] == row['recording_uid']]
            sess = sessions.loc[sessions['session_name'] == rec['session_name'].values[0]]
            exp = sess['experiment_name'].values[0]
            
            # if exp in self.naughty_experiments: continue
            # if exp not in self.good_experiments: continue

            tr = row['tracking_data']
            # checl_tracking_plotter(tr)
            print(row['recording_uid'], idx, ' of ', self.tracking.shape)
            in_rois = {}
            in_roi_tup = namedtuple('inroi', 'ins outs')
            print('  ... getting times in rois')
            roi_at_each_frame = tr[:, -1]
            for i, (roi, cc) in enumerate(self.rois.items()):
                # # Get time points in which mouse is in roi
                # x = line_smoother(tr[:, 0])
                # y = line_smoother(tr[:, 1])
                # in_x =np.where((cc.x0<=x) & (x<= cc.x1))
                # in_y = np.where((cc.y1<=y) & (y<= cc.y0))

                # in_xy = np.intersect1d(in_x, in_y)
                # in_xy = in_xy[in_xy>=9000]  # ? not sure what this line does
                # # raise NotImplementedError('Check what the line above does you dummy')

                in_roi = np.where(roi_at_each_frame==cc)[0]

                # get times at which it enters and exits
                xx = np.zeros(tr.shape[0])
                xx[in_roi] = 1
                enter_exit = np.diff(xx)
                enters, exits = np.where(np.diff(xx)>0)[0], np.where(np.diff(xx)<0)[0]
                in_rois[roi] = in_roi_tup(list(enters), list(exits))

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
                        gt = goodtime(sexit, tenter, texit, next_in, time_in_shelter, exp)
                        good_times.append(gt)



            # Check if trip includes trial and add to al trips dictionary
            print(' ... checking if trials')
            for good_time_n, g in enumerate(good_times):
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

                

                escape_start, escape_end = g.threat_exit -g.shelter_exit,  g.shelter_enter-g.shelter_exit
                # Ignore returns that are faster than 1s, probably errors
                if escape_end-escape_start <= 30*1: continue # ! hardcoded fps

                escape_rois_ids = np.trim_zeros(tracking_data[escape_start:escape_end, -2])
                
                try:
                    y_midpoint = np.where(tracking_data[escape_start:escape_end,1]>=550)[0][0]
                except:
                    print('smth went wrong')
                    continue
                
                escape_rois_id = escape_rois_ids[y_midpoint]
                if 22.0 == escape_rois_id:
                    arm_taken = 'Centre'
                elif escape_rois_id in [9.0, 5.0, 15.0, 14.0]:
                    arm_taken = 'Right_Far'
                elif escape_rois_id in [8.0, 2.0, 10.0, 11.0]:
                    arm_taken = 'Left_Far'
                elif escape_rois_id in [13.0, 6.0, 18.0]:
                    if 22.0 in escape_rois_ids:
                        arm_taken == "Centre"
                    elif 14.0 in escape_rois_ids[y_midpoint:y_midpoint+60] or 5.0 in escape_rois_ids[y_midpoint:y_midpoint+60]:
                        arm_taken = 'Right_Far'
                    else:
                        arm_taken = 'Right_Medium'

                elif escape_rois_id in [12.0, 3.0, 17.0]:
                    if 22.0 in escape_rois_ids:
                        arm_taken = 'Centre'
                    elif 11.0 in escape_rois_ids[y_midpoint:y_midpoint+60] or 2.0 in escape_rois_ids[y_midpoint:y_midpoint+60]:
                        arm_taken = 'Left_Far'
                    else: 
                        arm_taken = 'Left_Medium'
                else:
                    continue
                    # raise ValueError(escape_rois_id)

                # Manually checking arm of escape
                # print(arm_taken)
                # if 'Medium' in arm_taken.split('_'):
                #     f, ax = plt.subplots(facecolor=[.2, .2, .2])
                #     ax.plot(tracking_data[:, 0], tracking_data[:, 1], color='k', alpha=.5)
                #     ax.scatter(tracking_data[escape_start:escape_end, 0], tracking_data[escape_start:escape_end, 1], c=tracking_data[escape_start:escape_end, -2])
                #     ax.set(title=arm_taken)
                #     fld = 'C:\\Users\\Federico\\Desktop\\test'
                #     try:
                #         f.savefig(os.path.join(fld, '{}.png'.format(good_time_n)))
                #     except:
                #         pass
                #     # plt.close('all')

                #     plt.show()

                # escape_id = input("""
                #     Please Select arm of escape:
                #         - q: left far
                #         - w: left medium
                #         - e: centre
                #         - r: right medium
                #         - t: right far
                # """)


                # if escape_id: # if I don;t press anything it means it was correct
                #     input_lookup = dict(
                #         q='Left_Far',
                #         w='Left_Medium',
                #         e='Centre',
                #         r='Right_Medium',
                #         t='Right_Far'
                #     )
                #     try:
                #         arm_taken = input_lookup[escape_id]
                #     except:
                #         a = 1

                duration = (g.shelter_enter - g.threat_exit)/30 # ! hardcoded fps
                smooth_speed = line_smoother(tracking_data[:,2])
                max_speed = np.percentile(smooth_speed, 85)


                duration_lims = dict(Left_Far=9,
                                Left_Medium=6,
                                Centre=4,
                                Right_Medium=6,
                                Right_Far=9)

                if duration <= duration_lims[arm_taken]: # ! hardcoded arbritary variable
                    is_escape = 'true'
                else:
                    is_escape = 'false'


                self.all_trips.append(self.trip(g.shelter_exit, g.threat_enter, g.threat_exit, g.shelter_enter,
                            g.time_in_shelter, tracking_data, has_stim, row['recording_uid'], duration, max_speed, is_escape, arm_taken, g.experiment_name))
 

    def insert_trips_in_table(self):
        for i, trip in enumerate(self.all_trips): 
            key = trip._asdict()
            key['trip_id'] = i
            try:
                self.table.insert1(key)
                print('inserted: ', key['recording_uid'])
            except:
                print(' !!! - did not insert !!! - ', key['recording_uid'])



def check_table_inserts(table):
    data = pd.DataFrame(table.fetch())

    # data = data.loc[data['is_trial']=='true']
    
    # Plot XY traces sorted by arm taken
    arms = ['Left_Far', 'Left_Medium', 'Centre', 'Right_Medium', 'Right_Far']
    f, axarr = plt.subplots(3, 2, facecolor =[.2,  .2, .2])
    axarr = axarr.flatten()
    arr_size = 0
    colors =['r', 'b', 'm', 'g', 'y']
    for i, (arm, ax) in enumerate(zip(arms, axarr)):
        sel = data.loc[data['arm_taken'] == arm]
        for idx, row in sel.iterrows():
            t0, t1 = row['threat_exit']-row['shelter_exit'], row['shelter_enter']-row['shelter_exit']
            tracking = row['tracking_data']
            ax.scatter(tracking[t0:t0+150, 0], tracking[t0:t0+150, 1],c=tracking[t0:t0+150, -2],    s=1, alpha=.5)
            axarr[-1].scatter(tracking[t0:t0+150, 0], tracking[t0:t0+150, 1], c=colors[i],  s=1, alpha=.25)
            if tracking.shape[0] > arr_size: arr_size = tracking.shape[0]
        ax.set(title=arm, facecolor=[.2, .2, .2], xlim=[0, 1000], ylim=[200, 800])
    axarr[-1].set(facecolor=[.2, .2, .2])
        

    plt.show()




if __name__ == '__main__':
    # print('Ready')
    analyse_all_trips(erase_table=False, fill_in_table=True)

    print(AllTrips())
    check_table_inserts(AllTrips())

