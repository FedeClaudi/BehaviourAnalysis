import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from pandas.plotting import scatter_matrix
from collections import namedtuple
from itertools import combinations
from scipy.stats import gaussian_kde

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Binarizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DecTreC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# from sklearn.tree import export_graphvize

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
            self.plot = plot
            self.names = ['all', 'trials', 'not trials', 'fast not trials', 'all fasts', 'all slows']
            self.trips = pd.DataFrame(AllTrips.fetch())
            self.trials = self.trips.loc[self.trips['is_trial'] == 'true']
            self.not_trials = self.trips.loc[self.trips['is_trial'] == 'false']

            fast_returns_ids, all_fasts = self.get_velocities()
            all_slows = [i for i in self.trips['trip_id'] if i not in all_fasts]
            self.fast_returns = self.trips.loc[self.trips['trip_id'].isin(fast_returns_ids)]
            self.all_fasts = self.trips.loc[self.trips['trip_id'].isin(all_fasts)]
            self.all_slows = self.trips.loc[self.trips['trip_id'].isin(all_slows)]

            self.get_durations()
            self.analyse_roi_stay()
            self.analyse_return_path_length()

            self.x_displacement = self.get_x_displacement()
            self.trial_stats, self.in_shelter_stay = self.get_trial_stats(fast_returns_ids)

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
        """get_durations [get's the duration of each trip ]
        """

        # Get durations
        self.durations={}
        self.durations['all'] = self.calc_dur(self.trips['shelter_enter'], self.trips['threat_exit'])
        self.durations['trials'] = self.calc_dur(self.trials['shelter_enter'], self.trials['threat_exit'])
        self.durations['not trials'] = self.calc_dur(self.not_trials['shelter_enter'], self.not_trials['threat_exit'])
        self.durations['fast not trials'] = self.calc_dur(self.fast_returns['shelter_enter'], self.fast_returns['threat_exit'])
        self.durations['all fast'] = self.calc_dur(self.all_fast['shelter_enter'], self.all_fast['threat_exit'])
        self.durations['all slows'] = self.calc_dur(self.all_slows['shelter_enter'], self.all_slows['threat_exit'])

        self.plot_hist(self.durations, 'escape dur', nbins=500)

    def get_trial_stats(self, fast_returns_ids):
        stats = namedtuple('stats', 'is_trial time_in_shelter')
        names = ['all', 'trials', 'not trials', 'fast not trials']
        in_shelter_stay  = {n:[] for n in names}
        
        # Get trial or not for each return and time spent in shelter 
        trials = []
        for idx, row in self.trips.iterrows():
            in_shelter = row['time_in_shelter']/30
            in_shelter_stay['all'].append(in_shelter)
            
            if row['is_trial'] == 'true':
                trials.append(1)
                key = 'trials'
            else:
                trials.append(0)
                key = 'not trials'
                if row['trip_id'] in fast_returns_ids:
                    in_shelter_stay['fast not trials'].append(in_shelter)

            in_shelter_stay[key].append(in_shelter)

        self.plot_hist(in_shelter_stay, title='In shelter stay', xmax=300)

        # Return stats tuple
        return stats(trials, in_shelter_stay)

    def get_velocities(self):
        """
            get_velocities [get each velocity trace and the max vel percentiles for each trial]
        """
        self.velocities, self.vel_percentiles = {n:[] for n in self.names}, {n:[] for n in nself.ames}
        
        which_are_fast = [] # for each return,mark if it was fast or not

        # Get 
        fast_returns = []
        for idx, row in self.trips.iterrows():
            t0, t1 = row['threat_exit']-row['shelter_exit'], row['shelter_enter']-row['shelter_exit']
            vel = line_smoother(row['tracking_data'][t0:t1, 2])
            vel.flags.writeable = True
            vel[vel > 20] = 20
            perc = np.percentile(vel, 90)
            
            if perc>7:
                which_are_fast.append(1)
                self.velocities['all fasts'].append(vel)
                self.vel_percentiles['all fasts'].append(perc)
            else:
                which_are_fast.append(0)
                self.velocities['all slows'].append(vel)
                self.vel_percentiles['all slows'].append(perc)
            
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
        self.plot_hist(self.vel_percentiles, title = 'Velocity 75th percentile', 
                        xlabel='px/frame', nbins=50, plot_all=True)

        return fast_returns, which_are_fast

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


        if self.plot:
            sampl = np.random.uniform(low=0.0, high=10.0, size=(len(x_displacement),))
            f,ax = plt.subplots()
            ax.scatter(x_displacement, sampl, s=10, c='k', alpha=.5)
            ax.set(title='x displacement')

        return x_displacement

    def create_summary_dataframe(self):
        df_dict = {'duration':[], 'velocity':[], 'length':[], 'threat_stay':[], 'x_displacement':[], 'shelter_stay':[]}
        for i, duration in enumerate(self.durations['all']):
            df_dict['duration'].append(duration)
            df_dict['velocity'].append(self.vel_percentiles['all'][i])
            df_dict['length'].append(self.return_path_lengths['all'][i])
            df_dict['threat_stay'].append(self.threat_stay['all'][i])
            df_dict['x_displacement'].append(self.x_displacement[i])
            df_dict['shelter_stay'].append(self.in_shelter_stay['all'][i])
        df_dict['is trial'] = self.trial_stats
        
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
        if 'fast not trials' in var.keys():
            ax.hist(np.array(var['fast not trials']), bins=bins, color=[.4, .8, .4], alpha=.45, density=density, label='Fast Not trials')
        ax.set(facecolor=[.2, .2, .2], title=title, xlim=[0, xmax], xlabel=xlabel)
        ax.legend()

        # Plot by Fast vs Slow
        ax = axarr[3]
        if 'all fasts' in var.keys():
            ax.hist(np.array(var['all fasts']), bins=bins,  color='m', alpha=.75, density=density, label='Fast')
            ax.hist(np.array(var['all slows']), bins=bins,  color='y', alpha=.75, density=density, label='Slow')
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
        self.anonymous_data = self.data.copy()
        self.anonymous_data = self.anonymous_data.drop(['is trial','shelter_stay', 'threat_stay'], 1)
        # self.expand_data()


        self.inspect_data()
        self.pca_components = self.do_pca()
        self.clustered = self.kmeans()

        self.check_clustering()
        self.plot_points_density()

        plt.show()

    def expand_data(self):
        to_square = ['x_displacement', 'length', 'duration']
        for ts in to_square:
            squared = self.anonymous_data[ts].values
            self.anonymous_data['squared_'+ts] = pd.Series(np.square(squared))

    def inspect_data(self):
        self.anonymous_data.describe()
        self.anonymous_data.hist()
        self.corrr_mtx = self.anonymous_data.corr()

        for k in self.corrr_mtx.keys():
            print('\n Correlation mtx for {}'.format(k))
            print(self.corrr_mtx[k])

        scatter_matrix(self.anonymous_data, alpha=0.2, figsize=(6, 6), diagonal='kde')

        trials = self.anonymous_data.loc[self.data['is trial'] == 1]
        not_trials = self.anonymous_data.loc[self.data['is trial'] == 0]

        f, axarr = plt.subplots(5, 5, facecolor=[.8, .8, .8])
        axarr = axarr.flatten()
        combs = combinations(list(self.anonymous_data.columns), 2)
        counter = 0
        for i, (c1, c2) in enumerate(combs):
            if 'trial' in c1 or 'trial' in c2: continue
            ax = axarr[counter]
            counter += 1
            ax.set(facecolor=[.2, .2, .2], title='{}-{}'.format(c1, c2), xlabel=c1, ylabel=c2)
            ax.scatter(trials[c1].values, trials[c2].values, c=[.8, .2, .2], alpha=.2)
            ax.scatter(not_trials[c1].values, not_trials[c2].values, c=[.2, .2, .8], alpha=.2)
        f.tight_layout()
        # _, bins, _ = ax.hist(trials['threat_stay'].values, bins=100, alpha=.5)
        # ax.hist(not_trials['threat_stay'].values, bins=bins, alpha=.5)

        # plt.show()

    def plot_points_density(self):
        x = self.pca_components['principal component 1'].values
        y = self.pca_components['principal component 2'].values

        # Calculate the point density
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)

        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        fig, ax = plt.subplots(figsize=(5, 5), facecolor=[.2, .2, .2])
        ax.scatter(x, y, c=z, s=50, edgecolor='')
        ax.set(facecolor=[.8, .8, .8])

    def kmeans(self):  

        clustered_data = {}

        # create kmeans object
        f, axarr = plt.subplots(6, 1, facecolor=[.2, .2, .2])
        axarr=axarr.flatten()
        costs = []
        for c in range(2, 6):
            n_clusters=c
            ax = axarr[c-2]

            kmeans = KMeans(n_clusters=n_clusters)

            # fit kmeans object to data
            data = self.pca_components.drop(['is trial'], 1)
            kmeans.fit(data)

            # save new clusters for chart
            y_km = kmeans.fit_predict(self.anonymous_data)
            clustered_data[str(c)] = y_km

            # check results
            for i in range(n_clusters):
                t = data.loc[y_km == i]
                ax.scatter(t['principal component 1'], 
                    t['principal component 2'],
                    s=30, alpha=.2)
            ax.set(facecolor=[.2, .2, .2], title='{} Clusters'.format(c))

            interia = kmeans.inertia_
            print("k:",c, " cost:", round(interia))
            costs.append(round(interia))
        return clustered_data


        # Plot points

        # f, axarr = plt.subplots(len(self.data.columns), 1)
        # for i, k in enumerate(self.data.columns):
        #     if 'trial' in k: continue
        #     ax = axarr[i]
        #     _, bins, _ = ax.hist(trials[k].values, bins=100, alpha=.5)
        #     ax.hist(maybe_trials[k].values, bins=bins, alpha=.5)

    def check_clustering(self):
        f, axarr = plt.subplots(4, 1, facecolor=[.2, .2, .2])
        trials = self.data.loc[self.data['is trial'] == 1]

        for ax in axarr:
            ax.set(facecolor=[.2, .2, .2], title='velocity by cluster')
            _, bins, _ = ax.hist(trials['velocity'].values, bins=100, color=[.9, .9, .9], label='trials')

        for c, (k, v) in enumerate(self.clustered.items()):
            ax = axarr[c]
            for i in range(int(k)):
                t = self.data.loc[v == i]
                ax.hist(t['velocity'].values, bins=bins, label=str(i), alpha=.3)
        [ax.legend() for ax in axarr]

    def plot_pca(self, df):
        f, ax = plt.subplots(figsize=(16, 16), facecolor=[.2, .2, .2])
        d = dict(not_trials=(0, [.4, .4, .8], .4), trials=(1, [.8, .4, .4], .4),)

        ax.set(facecolor=[.2, .2, .2])
        for n, (i, c, a) in d.items():
            indicesToKeep = df['is trial'] == i
            ax.scatter(df.loc[indicesToKeep, 'principal component 1']
                , df.loc[indicesToKeep, 'principal component 2']
                , c = c, alpha=a, s = 30, label=n)
    
        # Plot a line
        # ax.plot([-2.5, 1], [2, -2], '--', color=[.4, .8, .4], linewidth=3)
        
        ax.legend()

    def do_pca(self):
        x = self.anonymous_data.values
        scaled = StandardScaler().fit_transform(x)
        
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(scaled)
        principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
        finalDf = pd.concat([principalDf, self.data['is trial']], axis = 1)

        print(pd.DataFrame(pca.components_,columns=self.anonymous_data.columns,index = ['PC-1','PC-2']))

        # Logistic Regression
        # _training_set, _test_set = train_test_split(finalDf.values, test_size=0.2, random_state=42)
        # training_set, training_labels = _training_set[:, :2], _training_set[:, -1]
        # test_set, test_labels = _test_set[:, :2], _test_set[:, :-1]
        # logisticRegr = LogisticRegression(solver = 'lbfgs')
        # logisticRegr.fit(training_set, training_labels)

        # predictions = logisticRegr.predict(test_set)
        # predictions = predictions.astype(int)
        # predictionsDf = pd.DataFrame(data=predictions, columns=['is trial'])
        # predictedDf = pd.concat([principalDf, predictionsDf['is trial']], axis = 1)

        # # print(logisticRegr.score(test_set.round(), test_labels.round()))

        # self.plot_pca(predictedDf)
        self.plot_pca(finalDf)

        return finalDf



if __name__ == '__main__':
    # analyse_all_trips(erase_table=True, fill_in_table=False, run_analysis=False)
    # analyse_all_trips(erase_table=False, fill_in_table=True, run_analysis=False)
    analyse_all_trips(erase_table=False, fill_in_table=False, run_analysis=True, plot=True)
    
    # cluster_returns()


