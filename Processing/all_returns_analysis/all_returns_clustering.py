import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
            'figure.figsize': (15, 15),
            'axes.labelsize': 'x-large',
            'axes.titlesize':'x-large',
            'xtick.labelsize':'x-large',
            'ytick.labelsize':'x-large',
            'font.size': 22}
pylab.rcParams.update(params)

import pandas as pd
from pandas.plotting import scatter_matrix
from collections import namedtuple
from itertools import combinations
from scipy.stats import gaussian_kde
import os
import seaborn as sn

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

from tsfresh import extract_relevant_features, extract_features
from tsfresh.utilities.dataframe_functions import impute

from Processing.tracking_stats.math_utils import line_smoother, calc_angle_between_points_of_vector
from Utilities.file_io.files_load_save import load_yaml
from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame

from Processing.all_returns_analysis.all_returns_database import *


class cluster_returns:
    def __init__(self):
        self.group_by = 'is trial'

        # Get and cleanup data
        analysis = analyse_all_trips()
        self.data = analysis.returns_summary
        self.anonymous_data = self.data.copy()
        self.anonymous_data = self.anonymous_data.drop(
            ['is trial', 'is fast', 'shelter_stay', 'threat_stay'], 1)
        
        # Features engineering and data minining
        self.expand_data()
        self.inspect_data()
        
        # Do PCA and K-means
        self.pca_components = self.do_pca()
        self.clustered = self.kmeans()

        # Plot stuff
        self.check_clustering()
        self.plot_points_density()

        plt.show()

    def expand_data(self):
        # to_square = ['x_displacement', 'length', 'duration']
        # for ts in to_square:
        #     squared = self.anonymous_data[ts].values
        #     self.anonymous_data['squared_'+ts] = pd.Series(np.square(squared))

        self.anonymous_data['dur_by_len'] = pd.Series(np.divide(
            self.anonymous_data['duration'].values, self.anonymous_data['length'].values))

    def inspect_data(self):
        self.anonymous_data.describe()
        self.anonymous_data.hist()
        self.corrr_mtx = self.anonymous_data.corr()

        for k in self.corrr_mtx.keys():
            print('\n Correlation mtx for {}'.format(k))
            print(self.corrr_mtx[k])

        n = len(self.anonymous_data.columns)
        scatter_matrix(self.anonymous_data, alpha=0.2,
                        figsize=(6, 6), diagonal='kde')

        trials = self.anonymous_data.loc[self.data[self.group_by] == 1]
        not_trials = self.anonymous_data.loc[self.data[self.group_by] == 0]

        f, axarr = plt.subplots(n, 4, facecolor=[.8, .8, .8])
        axarr = axarr.flatten()
        combs = combinations(list(self.anonymous_data.columns), 2)
        counter = 0
        for i, (c1, c2) in enumerate(combs):
            if 'trial' in c1 or 'trial' in c2:
                continue
            ax = axarr[counter]
            counter += 1
            ax.set(facecolor=[.2, .2, .2],
                title='{}-{}'.format(c1, c2), xlabel=c1, ylabel=c2)
            ax.scatter(trials[c1].values, trials[c2].values,
                    c=[.8, .2, .2], alpha=.2)
            ax.scatter(not_trials[c1].values,
                    not_trials[c2].values, c=[.2, .2, .8], alpha=.2)


    def plot_points_density(self):
        x = self.pca_components['principal component 1'].values
        y = self.pca_components['principal component 2'].values

        # Calculate the point density
        xy = np.vstack([x, y])
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
        axarr = axarr.flatten()
        costs = []
        for c in range(2, 6):
            n_clusters = c
            ax = axarr[c-2]

            kmeans = KMeans(n_clusters=n_clusters)

            # fit kmeans object to data
            data = self.pca_components.drop([self.group_by], 1)
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
            print("k:", c, " cost:", round(interia))
            costs.append(round(interia))
        return clustered_data

    def check_clustering(self):
        f, axarr = plt.subplots(4, 1, facecolor=[.2, .2, .2])
        trials = self.data.loc[self.data[self.group_by] == 1]

        for ax in axarr:
            ax.set(facecolor=[.2, .2, .2], title='velocity by cluster')
            _, bins, _ = ax.hist(
                trials['velocity'].values, bins=100, color=[.9, .9, .9], label='trials')

        for c, (k, v) in enumerate(self.clustered.items()):
            ax = axarr[c]
            for i in range(int(k)):
                t = self.data.loc[v == i]
                ax.hist(t['velocity'].values, bins=bins,
                        label=str(i), alpha=.3)
        [ax.legend() for ax in axarr]

    def plot_pca(self, df):
        f, ax = plt.subplots(facecolor=[.2, .2, .2])
        d = dict(not_trials=(0, [.4, .4, .8], .4),
                trials=(1, [.8, .4, .4], .4),)

        ax.set(facecolor=[.2, .2, .2])
        for n, (i, c, a) in d.items():
            indicesToKeep = df[self.group_by] == i
            ax.scatter(df.loc[indicesToKeep, 'principal component 1'],
                    df.loc[indicesToKeep, 'principal component 2'], c=c, alpha=a, s=30, label=n)

        # Plot a line
        # ax.plot([-2.5, 1], [2, -2], '--', color=[.4, .8, .4], linewidth=3)

        ax.legend()

    def do_pca(self):
        x = self.anonymous_data.values
        scaled = StandardScaler().fit_transform(x)

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(scaled)
        principalDf = pd.DataFrame(data=principalComponents, columns=[
                                'principal component 1', 'principal component 2'])
        finalDf = pd.concat([principalDf, self.data[self.group_by]], axis=1)

        print(pd.DataFrame(pca.components_,
                        columns=self.anonymous_data.columns, index=['PC-1', 'PC-2']))

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


class timeseries_returns:
    def __init__(self, load=False, trace=1):
        self.select_seconds = 20
        self.fps = 30
        self.n_clusters = 2
        self.sel_trace = trace
        self.save_plots= False

        analysis = analyse_all_trips()

        paths_names = ['Left_Far', 'Left_Medium', 'Centre', 'Right_Medium', 'Right_Far', 'all_paths']
        # paths_names = ['Right_Medium' ]    
        for self.path_n, self.path_name in enumerate(paths_names):
            if load:
                distance_mtx = np.load('Processing\\all_returns_analysis\\distance_mtx.npy')
            else:
                # Get prepped data
                self.data = analysis.returns_summary

                # Select returns - get tracking data
                self.data = self.prep_data()
                y, y_dict, y_list = self.get_y(self.data)
                
                # Get euclidean distance
                distance_mtx = self.distance(y)
            print('Got distance matrix')

            # Cluster 
            cluster_obj, self.data['cluster labels'] = self.cluster(distance_mtx)
            
            # Plot clusters
            # self.plot_combined_time_series()
            self.plot_all_heatmap()
            # self.plot_dendogram(distance_mtx)
            # self.plot_clusters_heatmaps()
            # self.plot_clusters_traces()

            # Multivariate Time Series Analysis
            # self.mvt_analysis()

    @staticmethod
    def convert_y_to_df(y):
        index = ['c{}'.format(i) for i in range(y.shape[1])]
        data = {idx:y[:,i] for i,idx in enumerate(index)}
        return pd.DataFrame.from_dict(data) 

    def prep_data(self):
        """prep_data [Select only returns along the R medium arm]
        """
        lims = dict(Left_Far=(-10000, -251),
                    Left_Medium=(-250, -100),
                    Centre=(-99, 99),
                    Right_Medium= (100, 250),
                    Right_Far= (251, 10000),
                    all_paths = (-10000, 10000))
        lm = lims[self.path_name]
        self.x_limits = lm
        new_data = self.data.loc[(self.data['x_displacement'] >= lm[0]) &
                                (self.data['x_displacement'] <= lm[1])]
        return new_data
    
    def get_y(self, arr, sel=None, smooth=False):
        length = self.select_seconds*self.fps
        y = np.zeros((length, arr.shape[0]))
        y_dict, y_list = {}, []
        if sel is None: sel = self.sel_trace

        for i, (idx, row) in enumerate(arr.iterrows()):
            t0, t_shelt = row['times']
            t1 = t0 + self.select_seconds*self.fps
            
            yy = np.array(np.round(row['tracking_data'][t0:t1, sel], 2), dtype=np.double)
            if sel == 2:
                yy[yy>20] = np.mean(yy)
            if smooth:
                yy = line_smoother(yy)
            y[:yy.shape[0], i] = yy
            y_dict[str(i)] = np.array(yy)
            y_list.append(np.array(yy))
        return y, y_dict, y_list

    @staticmethod
    def array_scaler(x):
        """Scales array to 0-1
        
        Dependencies:
            import numpy as np
        Args:
            x: mutable iterable array of float
        returns:
            scaled x
        """
        arr_min = np.min(x)
        x = np.array(x) - float(arr_min)
        arr_max = np.max(x)
        x = np.divide(x, float(arr_max))
        return x


    def array_sorter(self, y, mode='bigger', smooth=False):
        # Sort modes: bigger, lesser, maxval
        scaled = self.array_scaler(y)
        th = np.median(scaled)

        if self.sel_trace == 2:
            smooth = True
            mode = 'maxval'
        elif self.sel_trace == 4:
            # th = np.percentile(scaled, 75)
            th = 0.4
            mode = 'lesser'

        pos = []  # stores the index of the value of each column of Y at which the criteria are met (e.g. above th)
        for i in range(scaled.shape[1]):
            if smooth: l = line_smoother(scaled[:, i])
            else: l = scaled[:, i]

            try:
                if mode == 'bigger':
                    pos.append(np.where(l >= th)[0][0])
                elif mode == 'lesser':
                    pos.append(np.where(l <= th)[0][0])
                elif mode == 'maxval':
                    pos.append(np.where(l == np.max(l))[0][0])
                else: raise ValueError('unrecognised mode')
            except:
                pos.append(scaled.shape[1])

        sort_idx = np.argsort(pos)  # how to go from this to an ordered array
        return y[:, sort_idx[::-1]], sort_idx[::-1] # return the ordered array

    def distance(self, y):
        return euclidean_distances(y.T, y.T)

    def cluster(self, dist, plot=False):
        cluster = AgglomerativeClustering(n_clusters=self.n_clusters, affinity='euclidean', linkage='ward')  
        labels = cluster.fit_predict(dist)  
        
        if plot:
            f, axarr = plt.subplots(2, 1)
            axarr = axarr.flatten()
            for i in range(y.shape[1]):
                clst = labels[i]
                axarr[clst].plot(y[:, i], 'k', alpha=.5)  
        return cluster, labels

#######################################################################
#######################################################################
#######################################################################

    def plot_clusters_traces(self):
        clusters_ids = set(self.data['cluster labels'])

        f, axarr = plt.subplots(1, len(clusters_ids))

        for i, clust_id in enumerate(list(clusters_ids)[::-1]):
            selected = self.data.loc[self.data['cluster labels']==clust_id]
            x = self.get_y(selected, sel=0)
            y = self.get_y(selected, sel=1)
            v = self.get_y(selected, sel=2)

            axarr[i].scatter(x, y, c=v, cmap='inferno', s=10, alpha=.3)

    def plot_clusters_heatmaps(self):
        clusters_ids = set(self.data['cluster labels'])
        
        f, axarr = plt.subplots(2, len(clusters_ids))

        for _id in clusters_ids:
            selected = self.data.loc[self.data['cluster labels']==_id]
            y, y_dict, y_list = self.get_y(selected)

            axarr[0, _id].plot(y, color='k', alpha=.1)
            axarr[0, _id].plot(np.mean(y, 1), color='r', linewidth=3)
            sn.heatmap(y.T, ax=axarr[1, _id], )
            axarr[1, _id].set(title='{} - Cluster # {}'.format(self.path_name, _id))

    def plot_all_heatmap(self):
        cmap = 'inferno'
    
        if self.sel_trace == 1:
            vmax, vmin = 750, 350
        elif self.sel_trace == 2:
            vmax, vmin = 14, 0
        elif self.sel_trace == 4:
            vmax, vmin = 400, 50
        else:
            vmax, vmin = None, None

        y, _, _ = self.get_y(self.data)
        sort, idxs = self.array_sorter(y)
        y = np.fliplr(sort)

        v, _, _ = self.get_y(self.data, sel=2)
        v = v[:, idxs[::-1]]

        f, axarr  = plt.subplots(1, 2) 
        ax = axarr[0]
        sn.heatmap(y.T, ax=ax, cmap=cmap, xticklabels=False, vmax=vmax, vmin=vmin)
        ttls = ['', 'Y trace', 'V trace', 'Angle of mvmt', 'Distance from shelter']
        ax.set(title=self.path_name+' '+ttls[self.sel_trace])

        ax = axarr[1]
        sn.heatmap(v.T, ax=ax, cmap=cmap, xticklabels=False, vmax=15)
        ax.set(title='Velocity sorted')

        if self.save_plots:
            name = os.path.join('C:\\Users\\Federico\\Desktop',
                                self.path_name+' '+ttls[self.sel_trace]+'.png')
            f.savefig(name)

    def plot_dendogram(self, dist): 
        " plot the dendogram and the trace heatmaps divided by stimulus/spontaneous and cluster ID"
        print('Plotting...')

        # Create figure and axes
        f = plt.figure()
        clusters_ids = set(self.data['cluster labels'])    
        gs = gridspec.GridSpec(3, len(clusters_ids))
        dendo_ax = plt.subplot(gs[0, :])
        stim_axes = [plt.subplot(gs[1, i]) for i in range(len(clusters_ids))]
        spont_axes = [plt.subplot(gs[2, i]) for i in range(len(clusters_ids))]

        # Define some params for plotting
        if self.sel_trace == 1:
            center = 560
            cmap = 'inferno'
            vmax, vmin = 750, 350
            th = 500
        elif self.sel_trace == 2:
            center = 7
            cmap = 'inferno'
            vmax, vmin = 15, 2.5
            th = 4
        elif self.sel_trace == 4:
            center=None
            cmap = 'inferno'
            vmax, vmin = 400, 50
            th = 250
        else:
            center = None
            cmap = 'inferno'
            vmax, vmin = None, None
            th = 1

        # Plot dendogram
        ttls = ['', 'Y trace', 'V trace', 'Angle of mvmt', 'Distance from shelter']
        dend = shc.dendrogram(shc.linkage(dist, method='ward'), ax=dendo_ax, no_labels=True, truncate_mode = 'level', p=6) # , orientation='left')
        dendo_ax.set(title=self.path_name+' Clustered by : '+ttls[self.sel_trace])
        # Plot clusters heatmaps
        for i, clust_id in enumerate(list(clusters_ids)[::-1]):
            # Get data
            stim_evoked =  self.data.loc[(self.data['cluster labels']==clust_id)&(self.data['is trial']==1)]
            spontaneous =  self.data.loc[(self.data['cluster labels']==clust_id)&(self.data['is trial']==0)]
        
            stim_y, _, _ = self.get_y(stim_evoked)
            spont_y, _, _ = self.get_y(spontaneous)
            
            stim_y = np.fliplr(self.array_sorter(stim_y))
            spont_y = np.fliplr(self.array_sorter(spont_y))
        
            # Plot heatmaps
            if i == len(clusters_ids):
                show_cbar = True
            else:
                show_cbar = False

            try:
                sn.heatmap(stim_y.T, ax=stim_axes[i], center=center, cmap=cmap, 
                            xticklabels=False, vmax=vmax, vmin=vmin, cbar=True)
            except: pass
            sn.heatmap(spont_y.T, ax=spont_axes[i], center=center, cmap=cmap, 
                        xticklabels=False, vmax=vmax, vmin=vmin, cbar=True)

            # Set titles and stuff
            stim_axes[i].set(title="Stim. evoked - cluster {}".format(clust_id))
            spont_axes[i].set(title="Spontaneous - cluster {}".format(clust_id))

            # Save figure
            if self.save_plots:
                nm = self.path_name+' Clustered-'+ttls[self.sel_trace]
                name = os.path.join('C:\\Users\\Federico\\Desktop',
                                nm+'.png')
                f.savefig(name)

    def plot_combined_time_series(self):
        tst = namedtuple('timeseries', 'x y v d')
        clusters_ids = set(self.data['cluster labels'])
        colors = [[.7, .4, .4], [.4, .7, .4], [.4, .4, .7], 'm', 'y', 'k', 'w']

        f, axarr = plt.subplots(1, len(clusters_ids))
        for i, clust_id in enumerate(list(clusters_ids)[::-1]):
            selected =  self.data.loc[self.data['cluster labels']==clust_id]
            temp = [self.get_y(selected, sel=i, smooth=True) for i in [0, 1, 2, 4]]
            tss = tst(*[t[0] for t in temp])
            
            ax = axarr[i]
            ax.plot(tss.d, tss.v, color=colors[i], alpha=.2)
            ax.plot(np.mean(tss.d, 1), np.mean(tss.v, 1), color='k', alpha=1, linewidth=4)
            ax.set(facecolor=[.2, .2, .2], title='V/D clust: '+str(clust_id), xlabel='distnace', ylabel='velocity')


if __name__ == '__main__':
    """
        Traces IDs:
            > 0 - X
            > 1 - Y
            > 2 - V
            > 3 - Theta
            > 4 - distance from shelter
            > 5 - Body length
    """
    using_traces = [4, 5]
    for trace in using_traces:
        timeseries_returns(load=False, trace=trace)

    plt.show()
