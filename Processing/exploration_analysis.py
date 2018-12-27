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

    def __init__(self, erase_table=False, fill_in_table=False, run_analysis=True, plot=False):
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
            self.plot = plot

            self.trips = pd.DataFrame(AllTrips.fetch())
            self.trials = self.trips.loc[self.trips['is_trial'] == 'true']
            self.not_trials = self.trips.loc[self.trips['is_trial'] == 'false']

            fast_returns_ids = self.get_velocities()
            self.fast_returns = self.trips.loc[self.trips['trip_id'].isin(fast_returns_ids)]

            self.get_durations()
            self.analyse_roi_stay()
            self.analyse_return_path_length()

            if plot:
                # self.plot_all_trips()
                plt.show()

            self.returns_summary = self.create_summary_dataframe()

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
                x = line_smoother(tr[:, 0])
                y = line_smoother(tr[:, 1])
                in_x =np.where((cc.x0<=x) & (x<= cc.x1))
                in_y = np.where((cc.y1<=y) & (y<= cc.y0))
                # in_xy = [p for p in in_x[0] if p in in_y[0]]
                in_xy = np.intersect1d(in_x, in_y)
                in_xy = in_xy[in_xy>=9000]

                # get times at which it enters and exits
                xx = np.zeros(tr.shape[0])
                xx[in_xy] = 1
                enter_exit = np.diff(xx)
                enters, exits = np.where(np.diff(xx)>0)[0], np.where(np.diff(xx)<0)[0]
                in_rois[roi] = in_roi(list(enters), list(exits))
                
                # test_plot_rois_on_trace(x, y, self.rois, in_x, in_y,  when_in_rois[roi])


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
                    tenter = at_threat[0]
                    try:
                        texit = [t for t in in_rois['threat'].outs if t>tenter and t<next_in][-1]
                    except:
                        pass
                    else:
                        # if texit-tenter < 30: 
                        #     f, ax = plt.subplots()
                        #     ax.set(facecolor=[.2, .2, .2])
                        #     ax.plot(x, y, color=[.8, .8, .8], linewidth=1)
                        #     for roi, cc in self.rois.items():
                        #         rect = patches.Rectangle((cc.x0,cc.y0), cc.width,-cc.height,linewidth=1,edgecolor='b',facecolor='none', label=roi)
                        #         ax.add_patch(rect)  
                        #     ax.plot(x[sexit:next_in], y[sexit:next_in], 'g', linewidth=7)
                        #     for i, t in enumerate(at_threat):
                        #         if i == 0:
                        #             s=200
                        #         elif i == len(at_threat):
                        #             s=100
                        #         else:
                        #             s=50
                        #         ax.scatter(x[t], y[t], s=s, color='r')
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
    
    @staticmethod
    def calc_dur(t0, t1):
        """get_durations [calcs the number os seconds between t1 and t0]
        """
        durs = np.subtract(t0, t1)
        return np.divide(durs, 30) # <- convert frames to seconds for 30fps recordings 

    def get_durations(self):
        """get_durations [get's the duration of each trip ]
        """

        # Get durations
        self.durations={}
        self.durations['all'] = self.calc_dur(self.trips['shelter_enter'], self.trips['threat_exit'])
        self.durations['trials'] = self.calc_dur(self.trials['shelter_enter'], self.trials['threat_exit'])
        self.durations['not trials'] = self.calc_dur(self.not_trials['shelter_enter'], self.not_trials['threat_exit'])
        self.durations['fast not trials'] = self.calc_dur(self.fast_returns['shelter_enter'], self.fast_returns['threat_exit'])

        self.plot_hist(self.durations, 'escape dur', nbins=500)

    def get_velocities(self):
        """
            get_velocities [get each velocity trace and the max vel percentiles for each trial]
        """
        names = ['all', 'trials', 'not trials', 'fast not trials']
        self.velocities, self.vel_percentiles = {n:[] for n in names}, {n:[] for n in names}
        
        # Get 
        fast_returns = []
        for idx, row in self.trips.iterrows():
            vel = row['tracking_data'][row['threat_exit']-row['shelter_exit']:, 2]
            vel.flags.writeable = True
            vel[vel > 20] = 20
            perc = np.percentile(vel, 75)
            self.velocities['all'].append(vel)
            self.vel_percentiles['all'].append(perc)
    
            if row['is_trial'] == 'true':
                key = 'trials'
            else:
                if perc > 7: 
                    self.velocities['fast not trials'].append(vel)
                    self.vel_percentiles['fast not trials'].append(perc)                    
                    fast_returns.append(row['trip_id'])
                    
                key = 'not trials'
            self.velocities[key].append(vel)
            self.vel_percentiles[key].append(perc)
        self.plot_hist(self.vel_percentiles, title = 'Velocity 75th percentile', xlabel='px/frame', nbins=50)

        return fast_returns

    def analyse_return_path_length(self):
        self.return_path_lengths = {}
        for k, vels in self.velocities.items():
            self.return_path_lengths[k] = [np.sum(v) for v in vels] 

        self.plot_hist(self.return_path_lengths, title='return path length', xlabel='px', xmax=2000)

    def get_maze_components():
        """get_maze_components [get the maze component the mouse is on at each frame]
        """
        pass

    def analyse_roi_stay(self):
        self.threat_stay={}
        self.threat_stay['all'] = self.calc_dur(self.trips['threat_exit'], self.trips['threat_enter'])
        self.threat_stay['trials'] = self.calc_dur(self.trials['threat_exit'], self.trials['threat_enter'])
        self.threat_stay['not trials'] = self.calc_dur(self.not_trials['threat_exit'], self.not_trials['threat_enter'])
        self.threat_stay['fast not trials'] = self.calc_dur(self.fast_returns['threat_exit'], self.fast_returns['threat_enter'])

        self.plot_hist(self.threat_stay, title='in threat')

    def create_summary_dataframe(self):
        df_dict = {'duration':[], 'velocity':[], 'length':[], 'threat_stay':[]}
        for i, duration in enumerate(self.durations['all']):
            df_dict['duration'].append(duration)
            df_dict['velocity'].append(self.vel_percentiles['all'][i])
            df_dict['length'].append(self.return_path_lengths['all'][i])
            df_dict['threat_stay'].append(self.threat_stay['all'][i])

        return pd.DataFrame.from_dict(df_dict)

    #####################################################################
    #####################################################################
    #####################################################################

    def plot_hist(self, var, title='', nbins = 500,  xlabel='seconds', xmax=45, density=False):
        if not self.plot: 
            return
        f, ax = plt.subplots(facecolor=[.2, .2, .2])
        _, bins, _ = ax.hist(np.array(var['trials']), bins=nbins,  color=[.8, .4, .4], alpha=.75, density=density, label='Trials')
        ax.hist(np.array(var['not trials']), bins=bins, color=[.4, .4, .8], alpha=.75, density=density, label='Not trials')
        if 'fast not trials' in var.keys():
            ax.hist(np.array(var['fast not trials']), bins=bins, color=[.4, .8, .4], alpha=.55, density=density, label='Fast Not trials')
        ax.set(facecolor=[.2, .2, .2], title=title, xlim=[0, xmax], xlabel=xlabel)
        ax.legend()


    def plot_all_trips(self):
        c0,c1 = 0, 0
        f,axarr = plt.subplots(1, 2)
        for idx, row in self.trips.iterrows():
            if row['is_trial'] == 'true':
                key = 'trials'
                col = [.8, .4, .4]
                ax = axarr[0]
                c = c0
                c0 += 1
            else:
                key = 'not trials' 
                col = [.4, .4, .8]
                ax = axarr[1]
                c = c1
                c1 += 1
            
            x, y = line_smoother(row['tracking_data'][:, 0]), line_smoother(row['tracking_data'][:, 1])
            ax.plot(np.add(x,(300*(c%5))), np.add(y, (200*(c%10))), color=col, alpha=.5)



class cluster_returns:
    def __init__(self):
        analysis = analyse_all_trips()
        self.data = analysis.returns_summary  # data is a dataframe with all the escapes measurements

        self.inspect_data()

    def inspect_data(self):
        self.data.describe()
        self.data.hist()
        self.corrr_mtx = self.data.corr()

        for k in self.corrr_mtx.keys():
            print('\n Correlation mtx for {}'.format(k))
            print(self.corrr_mtx[k])

        plt.show()


if __name__ == '__main__':
    # analyse_all_trips(erase_table=True, fill_in_table=False, run_analysis=False)
    # analyse_all_trips(erase_table=False, fill_in_table=True, run_analysis=False)
    # analyse_all_trips(erase_table=False, fill_in_table=False, run_analysis=True)
    
    cluster_returns()


