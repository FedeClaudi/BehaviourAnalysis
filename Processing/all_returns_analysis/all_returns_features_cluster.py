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
        
        names, sel = ['d', 'v', 'o'], [4, 2, 3]
        
        self.prep_data()

    def prep_data(self,):
        # Get the 10 fastest and 10 slowest traces
        raw_data = {n:self.ts.get_y(sef.ts.data, sel=s) for n,s in zip(name, sel)}
        _, idxs = self.ts.array_sorter(raw_data['d'], sel=4)
        
        fast, slow = {}, {}
        for i, (n, raw) in enumerate(raw_data.items()):
            fast[n] = raw_data[n][idxs[:10]]
            slow[n] = raw_data[n][idxs[-10:]]

        self.fast = pd.DataFrame.from_dict(fast)
        self.slow = pd.DataFrame.from_dict(slow)

    def plotter_tester(self):
        f, axarr = plt.subplots(1, 3)
        for i, c in enumerate(list(self.fast.columns)):
            axarr[i].plot(self.fast[c].values, color='r', alpha=.75)
            axarr[i].plot(self.slow[c].values, color='k', alpha=.75)
            axarr[1].set(title=c)


if __name__ == "__main__":
    p = clusterer()
    # p.traces_plotter()


    plt.show()

