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

    def prep_data(self):
        def get_at_shelt(a):
            try:
                at_shelt = np.where(f <= 40)[0][0]
            except:
                return len(a)
            else:
                return at_shelt
        
        def get_trimmed_arrays(raw_data, idx):
            d = raw_data['d'][:, idx]
            at_shelt = get_at_shelt(d)

            temp = np.zeros((3, len(d)))
            temp[0, :at_shelt] = line_smoother_convolve(d[:at_shelt])
            temp[1, :at_shelt] = line_smoother_convolve(raw_data['v'][:, idx])
            temp[2, :at_shelt] = line_smoother_convolve(raw_data['o'][:, idx])
            return temp

        # Get the 10 fastest and 10 slowest traces
        names, sel = ['d', 'v', 'o'], [4, 2, 3]
        raw_data = {n:self.ts.get_y(self.ts.data, sel=s)[0] for n,s in zip(names, sel)}
        _, idxs = self.ts.array_sorter(raw_data['d'], sel=4)
        fast, slow = np.zeros((3, raw_data['d'].shape[0], 10)), []
        for i in range(10):
            fast[:, :, i] = get_trimmed_arrays(raw_data, idxs[i])
            slow[:, :, i] = get_trimmed_arrays(raw_data, idxs[-i])
            

        
        return fast, slow

    def plotter_tester(self):
        f, axarr = plt.subplots(1, 3)
        for i in range(3):
            axarr[i].plot(self.fast[i, :, :], color='r', alpha=.75)
            axarr[i].plot(self.slow[i, :, :], color='k', alpha=.75)


if __name__ == "__main__":
    p = clusterer()
    # p.traces_plotter()


    plt.show()

