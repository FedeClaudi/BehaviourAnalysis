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

from Processing.tracking_stats.math_utils import line_smoother
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
    def __init__(self):
        analysis = analyse_all_trips()
        # data is a dataframe with all the escapes measurements
        self.data = analysis.returns_summary

        self.get_r_returns()
        plt.show()

    def get_r_returns(self):
        def plotter(var, ttl=''):
            titles = ['x', 'y', 'xy', 'vel']
            ylims = [[400, 700], [300, 800], [350, 800], [0, 25]]
            f, axarr = plt.subplots(2, 2)
            axarr = axarr.flatten()
            for idx, row in var.iterrows():
                t0, t_shelt = row['times']
                x_t_shelt = t_shelt-t0

                t1 = t0 + 20*30
                axarr[0].plot(row['tracking_data'][t0:t1, 0], 'k', alpha=.2)
                axarr[1].plot(row['tracking_data'][t0:t1, 1], 'k', alpha=.2)
                axarr[2].plot(row['tracking_data'][t0:t1, 0],
                              row['tracking_data'][t0:t1, 1], 'k', alpha=.2)
                axarr[3].plot(line_smoother(
                    row['tracking_data'][t0:t1, 2]), 'k', alpha=.15)
                if x_t_shelt <= 20*30:
                    axarr[0].plot(x_t_shelt, row['tracking_data']
                                  [t_shelt, 0], 'o', color='r', alpha=.3)
                    axarr[1].plot(x_t_shelt, row['tracking_data']
                                  [t_shelt, 1], 'o', color='r', alpha=.3)
                    # axarr[3].plot(x_t_shelt, row['tracking_data'][t_shelt, 2], 'o', color='r', alpha=.3)
                axarr[2].plot(row['tracking_data'][t_shelt, 0],
                              row['tracking_data'][t_shelt, 1], 'o', color='r', alpha=.3)

            [ax.set(title=titles[i]+'  '+ttl, ylim=ylims[i])
             for i, ax in enumerate(axarr)]

        right_returns = self.data.loc[(self.data['x_displacement'] >= 100) & (
            self.data['x_displacement'] <= 150)]
        fast_right = right_returns.loc[right_returns['is fast'] == 1]
        slow_right = right_returns.loc[right_returns['is fast'] == 0]

        plotter(fast_right, 'fast')
        plotter(slow_right, 'slow')
        plotter(right_returns, 'ALL')


if __name__ == '__main__':
    #cluster_returns()

    timeseries_returns()