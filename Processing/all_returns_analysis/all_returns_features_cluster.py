import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
            'figure.figsize': (10, 3),
            'axes.titlesize':'x-large',
            'font.size': 22}
pylab.rcParams.update(params)

import pandas as pd
from pandas.plotting import scatter_matrix
from collections import namedtuple
from itertools import combinations
from scipy.stats import gaussian_kde
import os
import seaborn as sn

from scipy.signal import find_peaks

from Processing.tracking_stats.math_utils import *
from Processing.all_returns_analysis.all_returns_clustering import timeseries_returns

""" 
    Metrics:
        - duration
        - max (smoothed) velocity
        - orientation of mvmt variance
        - number of "turning points" of distance trace [ before reaching the shelter ]
"""


class clusterer:
    def __init__(self):
        # Get time series data
        self.ts = timeseries_returns(load=False, trace=4, do_all_arms=False)
        
        
        self.fast, self.slow = self.prep_data()
        self.plotter_tester()

    def prep_data(self,):
        # Get the 10 fastest and 10 slowest traces
        names, sel = ['d', 'v', 'o'], [4, 2, 3]
        raw_data = {n:self.ts.get_y(self.ts.data, sel=s)[0] for n,s in zip(names, sel)}
        raw_data['o'][raw_data['o'] < 1] = np.nan
        raw_data['o'][raw_data['o'] >359] = np.nan
        # x, y = self.ts.get_y(self.ts.data, sel=0)[0], self.ts.get_y(self.ts.data, sel=1)[0]
        # xy = np.array([x[:, 0], y])
        # raw_data['o'] = calc_angle_between_points_of_vector(xy.T)

        _, idxs = self.ts.array_sorter(raw_data['d'], sel=4)
        fast, slow = {}, {}
        for n, raw in raw_data.items():
            slow[n] = raw_data[n][:, idxs[:10]]
            fast[n] = raw_data[n][:, idxs[-10:]]
        return fast, slow

    def plotter_tester(self):
        f, axarr = plt.subplots(1, 3)
        for i, c in enumerate(list(self.fast.keys())):
            axarr[i].plot(self.fast[c], color='r', alpha=.75)
            axarr[i].plot(self.slow[c], color='k', alpha=.75)
            axarr[i].set(title=c)


if __name__ == "__main__":
    p = clusterer()
    # p.traces_plotter()


    plt.show()

