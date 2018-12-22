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

    def __init__(self, erase_table=False, fill_in_table=False, run_analysis=True):
        if erase_table:
            AllTrips.drop()

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
            print(self.table)

        if run_analysis:
            self.trips = pd.DataFrame(AllTrips.fetch())
            print(self.trips)
            self.inspect_durations()

    #####################################################################
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
            tr = row['tracking_data']
            print(row['recording_uid'], idx, ' of ', self.tracking.shape)

            # f,ax = plt.subplots()
            in_rois = {}
            in_roi = namedtuple('inroi', 'ins outs')

            print('  ... getting times in rois')
            for i, (roi, cc) in enumerate(self.rois.items()):
                # Get time points i which mouse is in roi
                in_x =np.where((cc.x0<=tr[:, 0]) & (tr[:, 0]<= cc.x1))
                in_y = np.where((cc.y1<=tr[:, 1]) & (tr[:, 1]<= cc.y0))
                # in_xy = [p for p in in_x[0] if p in in_y[0]]
                in_xy = np.intersect1d(in_x, in_y)
                in_xy = in_xy[in_xy>10000]

                # get times at which it enters and exits
                xx = np.zeros(tr.shape[0])
                xx[in_xy] = 1
                enter_exit = np.diff(xx)
                enters, exits = np.where(np.diff(xx)>0)[0], np.where(np.diff(xx)<0)[0]
                in_rois[roi] = in_roi(list(enters), list(exits))
                
                # test_plot_rois_on_trace(tr[:,0], tr[:, 1], self.rois, in_x, in_y,  when_in_rois[roi])


            # get complete s-t-s trips
            print(' ... getting good trips')
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
            print(' ... checking if trials')
            for g in good_times:
                rec_stimuli = self.stimuli.loc[self.stimuli['recording_uid'] == row['recording_uid']]
                stims_in_time = [s for s in rec_stimuli['stim_start'].values if g[0]<s<g[1]]
                if stims_in_time:
                    has_stim = 'true'
                else:
                    has_stim = 'false'
                # 'shelter_exit shelter_enter tracking_data is_trial recording_uid'
                self.all_trips.append(self.trip(g[0], g[1], tr[g[0]:g[1], :], has_stim, row['recording_uid']))

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
        for i, trip in enumerate(self.all_trips): 
            key = trip._asdict()
            key['trip_id'] = i
            try:
                self.table.insert1(key)
                print('inserted: ', key['recording_uid'])
            except:
                print(' !!! - did not intert !!! - ', key['recording_uid'])
            
    #####################################################################
    #####################################################################
    #####################################################################

    def inspect_durations(self):
        all_durations = np.subtract(self.trips['shelter_enter'].values, self.trips['shelter_exit'].values)
        all_durations = np.divide(all_durations, 30)

        only_trials = self.trips.loc[self.trips['is_trial'] == 'true']
        trials_durations = np.subtract(only_trials['shelter_enter'], only_trials['shelter_exit'])
        trials_durations = np.divide(trials_durations, 30)

        not_trials = self.trips.loc[self.trips['is_trial'] == 'false']
        not_trials_durations = np.subtract(not_trials['shelter_enter'], not_trials['shelter_exit'])
        not_trials_durations = np.divide(not_trials_durations, 30)


        f, ax = plt.subplots()
        ax.set(facecolor=[.2, .2, .2], title='Trips durations', ylabel='n', xlabel='s', xlim=[0, 150])
        _, bins, _ = ax.hist(all_durations, bins=200, color=[.8, .8, .8], label=None, alpha=0)
        ax.hist(trials_durations, bins=bins, color=[.8, .4, .4], alpha=.5, label='trials')
        ax.hist(not_trials_durations, bins=bins, color=[.4, .4, .8], alpha=.5, label='not trials')
        ax.plot([np.median(trials_durations), np.median(trials_durations)], [0, 60], color=[.8, .4, .4], linewidth=2.5)
        ax.plot([np.median(not_trials_durations), np.median(not_trials_durations)], [0, 60], color=[.4, .4, .8], linewidth=2.5)
        ax.legend()

        # f2, axarr = plt.subplots(1, 2)
        # for trip in only_trials['tracking_data'].values:
        #     axarr[0].plot(trip[:, 0], trip[:, 1], color=[.8, .4, .4])
        #     axarr[0].set(title='trials', facecolor = [.2, .2, .2])
        # for trip in not_trials['tracking_data'].values:
        #     axarr[1].plot(trip[:, 0], trip[:, 1], color=[.8, .4, .4])
        #     axarr[1].set(title='not trials', facecolor = [.2, .2, .2])

        plt.show()

if __name__ == '__main__':
    # AllTrips.drop()
    
    analyse_all_trips(erase_table=False, fill_in_table=False, run_analysis=True)

