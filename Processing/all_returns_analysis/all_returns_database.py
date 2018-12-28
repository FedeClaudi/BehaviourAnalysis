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
            self.trip = namedtuple('trip', 'shelter_exit threat_enter threat_exit shelter_enter time_in_shelter tracking_data is_trial recording_uid')
            self.get_trips()

            # Insert good trips into trips table
            self.table = AllTrips()
            self.insert_trips_in_table()

            print(self.table)

        if run_analysis:
            # set Vars
            self.plot = plot
            self.names = ['all', 'trials', 'not_trials', 'fast_not_trials', 'all_fasts', 'all_slows']
            
            # Get all trips and split between trials and not trials
            self.trips = pd.DataFrame(AllTrips.fetch())
            self.trials = self.trips.loc[self.trips['is_trial'] == 'true']
            self.not_trials = self.trips.loc[self.trips['is_trial'] == 'false']

            # Get velocity features and split between fast and slow 
            fast_returns_ids, all_fasts = self.get_velocities()
            all_slows = [i for i in self.trips['trip_id'] if i not in all_fasts]
            self.fast_returns = self.trips.loc[self.trips['trip_id'].isin(fast_returns_ids)]
            self.all_fasts = self.trips.loc[self.trips['trip_id'].isin(all_fasts)]
            self.all_slows = self.trips.loc[self.trips['trip_id'].isin(all_slows)]

            # Group all dataframes
            self.all_dfs = dict(
                    all=self.trips,
                    trials=self.trials,
                    not_trials=self.not_trials, 
                    fast_not_trials=self.fast_returns,
                    all_fasts=self.all_fasts,
                    all_slows=self.all_slows,
            )

            self.get_durations()
            self.analyse_roi_stay()
            self.analyse_return_path_length()

            self.x_displacement = self.get_x_displacement()
            self.trial_stats, self.in_shelter_stay, self.fast_stats = self.get_trial_stats(fast_returns_ids,all_fasts)

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

            # f, ax = plt.subplots()
            # ax.plot(x, y, 'k', alpha=.5)
            # ax.plot(x[inx], y[inx], 'r',alpha=.75, label='x')
            # ax.plot(x[iny], y[iny], 'b',alpha=.75, label='y')
            # ax.plot(x[inboth], y[inboth], 'r')
            # ax.legend()
            # plt.show()

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
                        good_times.append((sexit, tenter, texit, next_in, time_in_shelter))

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
                self.all_trips.append(self.trip(g[0], g[1], g[2], g[3], g[4], tr[g[0]:g[4]+g[3], :], has_stim, row['recording_uid']))

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

    @staticmethod
    def scale_data(d):
        scaler = StandardScaler()
        v = np.asarray(d).reshape((-1, 1))
        scaled = scaler.fit_transform(v.astype(np.float64))
        return scaled.flatten()

    def get_durations(self):
        # Get durations
        self.durations={k:self.calc_dur(v['shelter_enter'], v['threat_exit'])}
        # Plot
        self.plot_hist(self.durations, 'escape dur', nbins=500)

    def get_trial_stats(self, fast_returns_ids, all_fasts_ids):
        stats = namedtuple('stats', 'is_trial time_in_shelter if_fast')
        in_shelter_stay  = {n:[] for n in self.names}
        
        # Get trial or not for each return and time spent in shelter 
        trials, fastones = [], []
        for idx, row in self.trips.iterrows():
            in_shelter = row['time_in_shelter']/30
            in_shelter_stay['all'].append(in_shelter)
            # Check if its a trial
            if row['is_trial'] == 'true':
                trials.append(1)
                key = 'trials'
            else:
                trials.append(0)
                key = 'not trials'
                if row['trip_id'] in fast_returns_ids:
                    in_shelter_stay['fast not trials'].append(in_shelter)

            # Check if its a fast one
            if row['trip_id'] in all_fasts_ids:
                key2 = 'all fasts'
                fastones.append(1)
            else:
                key2 = 'all slows'
                fastones.append(0)

            in_shelter_stay[key].append(in_shelter)
            in_shelter_stay[key2].append(in_shelter)

        self.plot_hist(in_shelter_stay, title='In shelter stay', xmax=300)

        # Return stats tuple
        return stats(trials, in_shelter_stay, fastones)

    def get_velocities(self):
        velocity_th, velocity_percentile, percentile_th = 20, 90, 7
        self.velocities, self.vel_percentiles = {n:[] for n in self.names}, {n:[] for n in self.names} 
        which_are_fast = [] # for each return,mark if it was fast or not

        # Get 
        fast_returns = []
        for idx, row in self.trips.iterrows():
            # Get velocity and percentile
            t0, t1 = row['threat_exit']-row['shelter_exit'], row['shelter_enter']-row['shelter_exit']
            vel = line_smoother(row['tracking_data'][t0:t1, 2])
            vel.flags.writeable = True
            if np.any(vel[vel > velocity_th]):
                self.velocities['all fasts'].append(None)
                self.vel_percentiles['all fasts'].append(None)
            perc = np.percentile(vel, velocity_percentile)
            
            # Check if its a fast return
            if perc>percentile_th:
                which_are_fast.append(row['trip_id'])
                self.velocities['all fasts'].append(vel)
                self.vel_percentiles['all fasts'].append(perc)
            else:
                which_are_fast.append(0)
                self.velocities['all slows'].append(vel)
                self.vel_percentiles['all slows'].append(perc)
            
            # Check if its a trial
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
        # Plot
        self.plot_hist(self.vel_percentiles, title = 'Velocity {}th percentile'.format(velocity_percentile), 
                        xlabel='px/frame', nbins=50, plot_all=True)
        return fast_returns, which_are_fast

    def analyse_return_path_length(self):
        self.return_path_lengths = {}
        for k, vels in self.velocities.items():
            self.return_path_lengths[k] = [np.sum(v) for v in vels] 
        self.plot_hist(self.return_path_lengths, title='return path length', xlabel='px', xmax=2000)

    def analyse_roi_stay(self):
        self.threat_stay={k:self.calc_dur(v['threat_exit'], v['threat_enter'])}
        self.plot_hist(self.threat_stay, title='in threat')

    def get_x_displacement(self):
        x_displacement = []
        for idx, row in self.trips.iterrows():
            # Get x between the time T is left and the mouse is at the shelter
            t0, t1 = row['threat_exit']-row['shelter_exit'], row['shelter_enter']-row['shelter_exit']
            x = line_smoother(np.add(row['tracking_data'][t0:t1, 0], -500)) # center on X
            y = line_smoother(row['tracking_data'][t0:t1, 1])

            # Get left-most and right-most points
            left_most, right_most = min(x), max(x)
            if abs(left_most)>=abs(right_most):
                tokeep = left_most
            else:
                tokeep = right_most
            x_displacement.append(tokeep)

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

    def create_summary_dataframe(self):
        df_dict = {'duration':[], 'velocity':[], 'length':[], 'threat_stay':[], 'x_displacement':[], 'shelter_stay':[], 'times':[]}
        for i, duration in enumerate(self.durations['all']):
            
            row = self.trips.loc[i]
            t0, t1 = row['threat_exit']-row['shelter_exit'], row['shelter_enter']-row['shelter_exit']
            df_dict['times'].append((t0, t1))
            df_dict['duration'].append(duration)
            df_dict['velocity'].append(self.vel_percentiles['all'][i])
            df_dict['length'].append(self.return_path_lengths['all'][i])
            df_dict['threat_stay'].append(self.threat_stay['all'][i])
            df_dict['x_displacement'].append(self.x_displacement[i])
            df_dict['shelter_stay'].append(self.in_shelter_stay['all'][i])
        df_dict['is trial'] = self.trial_stats
        df_dict['is fast'] = self.fast_stats
        df_dict['tracking_data'] = self.trips['tracking_data']
        
        # scale
        # scaled_df_dict = {k: self.scale_data(np.array(v)) for k,v in df_dict.items()}
        return pd.DataFrame.from_dict(df_dict)

    #####################################################################
    #####################################################################
    #####################################################################

    def plot_hist(self, var, title='', nbins=250,  xlabel='seconds',
                xmax=45, density=False, plot_all=False):
        if not self.plot: 
            return
        
        f, axarr = plt.subplots(2, 2, facecolor=[.2, .2, .2])
        axarr = axarr.flatten()
        for ax in axarr:
            ax.set(facecolor=[.2, .2, .2], title=title, xlim=[0, xmax], xlabel=xlabel)

        # Plot all 
        axarr[0].set(facecolor=[.2, .2, .2], title =  title+' all')
        _, nbins, _ = axarr[0].hist(np.array(var['all']), bins=nbins, color=[.8, .8, .8], alpha=1, density=density, label='Fast Not trials')

        # Plot by trial vs no trial
        ax = axarr[2]
        _, bins, _ = ax.hist(np.array(var['trials']), bins=nbins,  color=[.8, .4, .4], alpha=.75, density=density, label='Trials')
        ax.hist(np.array(var['not trials']), bins=bins, color=[.4, .4, .8], alpha=.55, density=density, label='Not trials')
        if 'fast_not_trials' in var.keys():
            ax.hist(np.array(var['fast_not_trials']), bins=bins, color=[
                    .4, .8, .4], alpha=.45, density=density, label='Fast Not trials')
        ax.set(facecolor=[.2, .2, .2], title=title, xlim=[0, xmax], xlabel=xlabel)
        ax.legend()

        # Plot by Fast vs Slow
        ax = axarr[3]
        if 'all_fasts' in var.keys():
            ax.hist(np.array(var['all_fasts']), bins=bins,  color='m', alpha=.75, density=density, label='Fast')
            ax.hist(np.array(var['all_slows']), bins=bins,  color='y', alpha=.75, density=density, label='Slow')
            ax.legend()


if __name__ == '__main__':
    # analyse_all_trips(erase_table=True, fill_in_table=False, run_analysis=False)
    # analyse_all_trips(erase_table=False, fill_in_table=True, run_analysis=False)
    # analyse_all_trips(erase_table=False, fill_in_table=False, run_analysis=True, plot=True)
    
    #cluster_returns()
    
    timeseries_returns()


