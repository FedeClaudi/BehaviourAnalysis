import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from collections import namedtuple

from database.NewTablesDefinitions import *
from database.dj_config import start_connection

from Processing.tracking_stats.math_utils import line_smoother
from Utilities.file_io.files_load_save import load_yaml
from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame


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
            self.trip = namedtuple('trip', 'shelter_exit threat_enter threat_exit shelter_enter tracking_data is_trial recording_uid')
            self.get_trips()

            # Insert good trips into trips table
            self.table = AllTrips()
            self.insert_trips_in_table()
            print(self.table)

        if run_analysis:
            self.trips = pd.DataFrame(AllTrips.fetch())
            print(self.trips)
            self.trials = self.trips.loc[self.trips['is_trial'] == 'true']
            self.not_trials = self.trips.loc[self.trips['is_trial'] == 'false']

            self.get_durations()
            self.get_velocities()
            self.analyse_roi_stay()

            plt.show()

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
                in_xy = in_xy[in_xy>=9000]

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
                at_threat = [i for i in in_rois['threat'].ins if i > sexit and i < next_in]
                
                if at_threat:
                    texit = at_threat[-1]
                    tenter = at_threat[0]
                    good_times.append((sexit, tenter, texit, next_in))

            # Check if trip includes trial and add to al trips dictionary
            print(' ... checking if trials')
            for g in good_times:
                rec_stimuli = self.stimuli.loc[self.stimuli['recording_uid'] == row['recording_uid']]
                stims_in_time = [s for s in rec_stimuli['stim_start'].values if g[0]<s<g[2]]
                if stims_in_time:
                    has_stim = 'true'
                else:
                    has_stim = 'false'

                # shelter_exit threat_enter threat_exit shelter_enter tracking_data is_trial recording_uid
                self.all_trips.append(self.trip(g[0], g[1], g[2], g[3], tr[g[0]:g[3], :], has_stim, row['recording_uid']))

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

    def get_durations(self):
        """get_durations [get's the duration of each trip ]
        """
        def calc(t0, t1):
            """get_durations [calcs the number os seconds between t1 and t0]
            """
            durs = np.subtract(t1, t0)
            return np.divide(durs, 30) # <- convert frames to seconds for 30fps recordings 

        # Get durations
        self.durations={}
        self.durations['all'] = calc(self.trips['shelter_enter'], self.trips['threat_exit'])
        self.durations['trials'] = calc(self.trials['shelter_enter'], self.trials['threat_exit'])
        self.durations['not trials'] = calc(self.not_trials['shelter_enter'], self.not_trials['threat_exit'])

    def get_velocities(self):
        """
            get_velocities [get each velocity trace and the max vel percentiles for each trial]
        """
        names = ['all', 'trials', 'not trials']
        self.velocities, self.vel_percentiles = {n:[] for n in names}, {n:[] for n in names}
        
        # Get velocities
        for idx, row in self.trips.iterrows():
            vel = row['tracking_data'][:, 2]
            raise NotImplementedError('Adjust velocity to just get escape')
            self.velocities['all'].append(vel)
            self.vel_percentiles['all'].append(np.percentile(vel, 75))
    
            if row['is_trial'] == 'true':
                key = 'trials'
            else:
                key = 'not trials'
            self.velocities[key].append(vel)
            self.vel_percentiles[key].append(np.percentile(vel, 75))

        self.plot_hist(self.vel_percentiles)

    def get_maze_components():
        """get_maze_components [get the maze component the mouse is on at each frame]
        """
        pass

    def analyse_roi_stay(self):
        names = ['all', 'trials', 'not trials']
        self.threat_stay = {n:[] for n in names}

        for idx, row in self.trips.iterrows():
            dur = row['threat_exit'] - row['threat_enter']
            self.threat_stay['all'] = dur
            if row['is_trial'] == 'true':
                key = 'trials'
            else:
                key = 'not trials'
            self.threat_stay[key] = dur

        self.plot_hist(self.threat_stay)

    def plot_hist(self, var, title=''):
        f, ax = plt.subplots(facecolor=[.2, .2, .2])
        _, bins, _ = ax.hist(np.array(var['trials']), color=[.8, .2, .2], alpha=.5, label='Trials')
        ax.hist(np.array(var['not trials']), bins=bins, color=[.2, .2, .8], alpha=.5, label='Not trials')
        ax.set(facecolor=[.2, .2, .2], titile=title)
        ax.legend()


if __name__ == '__main__':
    analyse_all_trips(erase_table=False, fill_in_table=False, run_analysis=True)
    

    # Plot
    # f, axarr = plt.subplots(1, 2, facecolor=[.6, .6, .6])
    # ax = axarr[0]
    # ax.set(facecolor=[.2, .2, .2], title='Trips durations', ylabel='n', xlabel='s', xlim=[0, 60], ylim=[0, 80])
    # _, bins, _ = ax.hist(all_durations, bins=200, color=[.8, .8, .8], label=None, alpha=0)
    # ax.hist(trials_durations, bins=bins, color=[.8, .4, .4], alpha=.5, label='trials')
    # ax.hist(not_trials_durations, bins=bins, color=[.4, .4, .8], alpha=.5, label='not trials')
    # ax.plot([np.median(trials_durations), np.median(trials_durations)], [0, 60], color=[.8, .4, .4], linewidth=2.5)
    # ax.plot([np.median(not_trials_durations), np.median(not_trials_durations)], [0, 60], color=[.4, .4, .8], linewidth=2.5)
    # ax.legend()

    # ax2 = axarr[1]
    # ax2.set(facecolor=[.2, .2, .2], title='max vel', ylabel='n', xlabel='arbritary unit', xlim=[0, 20], ylim=[0, 100])
    # _, bins, _ = ax2.hist(all_vels, bins=50, color=[.8, .8, .8], label=None, alpha=0)
    # ax2.hist(trials_vels, bins=bins, color=[.8, .4, .4], alpha=.5, label='trials')
    # ax2.hist(not_trials_vels, bins=bins, color=[.4, .4, .8], alpha=.5, label='not trials')
    # ax2.plot([np.median(trials_vels), np.median(trials_vels)], [0, 30], color=[.8, .4, .4], linewidth=2.5)
    # ax2.plot([np.median(not_trials_vels), np.median(not_trials_vels)], [0, 30], color=[.4, .4, .8], linewidth=2.5)
    # ax2.legend()

    # f2, ax3 = plt.subplots(facecolor=[.2, .2, .2])
    # for idx, row in self.trips.iterrows():
    #     x = row['tracking_data'][:, 0]
    #     y = row['tracking_data'][:, 1]
    #     v = row['tracking_data'][:, 2]
    #     v.setflags(write=1)
    #     v[v>30] = 30
    #     ax3.scatter(x, y, c=v, s=5, cmap='Oranges')
    # ax3.set(facecolor=[.2, .2, .2])

    # f3, axarr = plt.subplots(1, 2, facecolor=[.2, .2, .2])

    # trials_vels, not_trials_vels = [], []
    # for idx, row in trials.iterrows():
    #     v = row['tracking_data'][:, 2]
    #     v.setflags(write=1)
    #     v[v>30] = 30
    #     lsv = line_smoother(v, order=11)
    #     axarr[0].plot(lsv, linewidth=.5, color=[.8, .4, .4], alpha=.75)
    # axarr[0].set(title='trials', facecolor=[.2,.2, .2])
    # for idx, row in not_trials.iterrows():
    #     v = row['tracking_data'][:, 2]
    #     v.setflags(write=1)
    #     v[v>30] = 30
    #     lsv = line_smoother(v, order=11)
    #     axarr[1].plot(lsv, linewidth=.5, color=[.4, .4, .8], alpha=.75)
    # axarr[1].set(title='not trials', facecolor=[.2,.2, .2])
    # axarr[0].plot(np.mean(trials_vels))
    # axarr[1].plot(np.mean(not_trials_vels))
