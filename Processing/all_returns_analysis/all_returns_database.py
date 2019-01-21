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
        roi = namedtuple('roi', 'x0 x1 width y0 y1 height')
        formatted_rois = {}
        rois_dict = load_yaml('Utilities\\video_and_plotting\\template_points.yml')

        for roi_name, cc in rois_dict.items():
            if roi_name == 'translators': continue
            formatted_rois[roi_name] = roi(cc['x'], cc['x']+cc['width'], cc['width'],
                                            cc['y'], cc['y']-cc['height'], cc['height'])
        return formatted_rois

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
            checl_tracking_plotter(tr)
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
                        gt = goodtime(sexit, tenter, texit, next_in, time_in_shelter, exp)
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
                max_speed = np.percentile(smooth_speed, 85)


                x_displacement = self.get_x_displacement(tracking_data[:, 0], tracking_data[:, 1], g.threat_exit, g.shelter_exit, g.shelter_enter)
                arms_lims = dict(Left_Far=(-10000, -251),
                                Left_Medium=(-250, -100),
                                Centre=(-99, 69),
                                Right_Medium= (70, 250),
                                Right_Far= (251, 10000))

                for k, (x0, x1) in arms_lims.items():
                    if x0 <= x_displacement <= x1:
                        arm_taken = k


                duration_lims = dict(Left_Far=12,
                                Left_Medium=9,
                                Centre=4,
                                Right_Medium=9,
                                Right_Far=12)

                if duration <= duration_lims[arm_taken]: # ! hardcoded arbritary variable
                    is_escape = 'true'
                else:
                    is_escape = 'false'


                self.all_trips.append(self.trip(g.shelter_exit, g.threat_enter, g.threat_exit, g.shelter_enter,
                            g.time_in_shelter, tracking_data, has_stim, row['recording_uid'], duration, max_speed, is_escape, arm_taken, g.experiment_name))
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



def check_table_inserts(table):
    # Plot XY traces based on arm
    # Plot D traces sorted by tur
    #    sorted by is trial
    #    sorted by maxV
    #    # sorted by is escape
    data = pd.DataFrame(table.fetch())
    
    # Plot XY traces sorted by arm taken
    arms = ['Left_Far', 'Left_Medium', 'Centre', 'Right_Medium', 'Right_Far']
    f, axarr = plt.subplots(3, 2, facecolor =[.2,  .2, .2])
    axarr = axarr.flatten()
    arr_size = 0
    for arm, ax in zip(arms, axarr):
        sel = data.loc[data['arm_taken'] == arm]
        for idx, row in sel.iterrows():
            t0, t1 = row['threat_exit']-row['shelter_exit'], row['shelter_enter']-row['shelter_exit']
            tracking = row['tracking_data']
            ax.scatter(tracking[t0:t1, 0], tracking[t0:t1, 1], color=[.8, .8, .8], s=5, alpha=.5)
            if tracking.shape[0] > arr_size: arr_size = tracking.shape[0]
        ax.set(title=arm, facecolor=[.2, .2, .2], xlim=[0, 1000], ylim=[200, 800])

    # Prep to plot sorted D traces
    d = np.zeros((arr_size, data.shape[0]))
    for idx, row in data.iterrows():
        temp = row['tracking_data'][:, 4]
        d[:len(temp), idx] = temp

    # Plot
    f, axarr = plt.subplots(2, 2,  facecolor=[.2, .2, .2])
    axarr = axarr.flatten()

    sort_by_mvel = np.argsort(data['max_speed'].values)
    sort_by_dur = np.argsort(data['duration'].values)
    for i in range(len(sort_by_mvel)):
        # axarr[0].plot(np.add(d[:, sort_by_mvel[i]], i*700))
        # axarr[1].plot(np.add(d[:, sort_by_dur[i]], i*700))
        t0 = data['threat_exit'].values[i] - data['shelter_exit'].values[i]
        if data['is_trial'].values[i] == 'true':
            trial_col = 'r'
            trial_ax = 0
        else:
            trial_col = 'w'
            trial_ax = 2

        if data['is_escape'].values[i] == 'true':
            escape_col = 'g'
            escape_ax = 1
        else:
            escape_col = 'w'
            escape_ax = 3

        axarr[trial_ax].plot(line_smoother(d[t0:, i]), color=trial_col, alpha=.2, linewidth=.5)
        axarr[escape_ax].plot(line_smoother(d[t0:, i]), color=escape_col, alpha=.2, linewidth=.5)

    titles = ['D trial', 'D escape', 'D spontaneous', 'D not escape']
    for i, tit in enumerate(titles):
        axarr[i].set(title=tit, facecolor=[.2, .2, .2], xlim=[0, 1000])

    f, ax = plt.subplots(facecolor=[.2,  .2, .2])
    dur = np.sort(data['duration'].values)
    speed_sort = np.argsort(data['duration'].values)
    sped = data['max_speed'].values[speed_sort]
    ax.plot(dur, 'o', color='r', label='duration', alpha=.8)
    ax.plot(sped,'o', color='g', label='speed', alpha=.8)
    ax.set(facecolor=[.2, .2, .2])
    ax.legend()
    plt.show()




if __name__ == '__main__':
    print('Ready')
    analyse_all_trips(erase_table=False, fill_in_table=True)

    print(AllTrips())
    # check_table_inserts(AllTrips())

