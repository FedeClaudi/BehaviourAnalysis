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

# write a function that plots all traces (e.g. distance) sorted by smth (e.g. max velocity) with and witouth overlay and 
# color coded by stimulus/spontaneous

class features_finder:
    def __init__(self, y):
        return
        
    def make_df(self):
        time_to_th = self.time_to_th(y, 40)
        _min, _max, time_to_min, time_to_max = self.get_max_min(y)
        npeaks, ntrufts = self.get_peaks_trufts(y)

        dct = dict(
            time_to_th=time_to_th,
            _min=_min,
            _max=_max,
            time_to_min=time_to_min,
            time_to_max=time_to_max,
            npeas=npeaks,
            ntrufts=ntrufts
        )

        return pd.DataFrame.from_dict(dct)

    @staticmethod
    def time_to_th(y, th):
        times = []
        for i in range(y.shape[1]):
            try:
                t = np.where(y[:, i] <= th)[0][0]
            except:
                t = y.shape[0]
            times.append(t)
        return times

    @staticmethod
    def get_max_min(y):
        _min, _max, time_to_min, time_to_max = [], [], [], []
        for i in range(y.shape[1]):
            _min.append(np.min(y[:, i]))
            _max.append(np.max(y[:, i]))
            time_to_min.append(np.where(y[:, i]==_min[-1])[0][0])
            time_to_max.append(np.where(y[:, i]==_max[-1])[0][0])
        return _min, _max, time_to_min, time_to_max

    @staticmethod
    def get_peaks_trufts(y):
        npeaks, ntrufts = [], []
        for i in range(y.shape[1]):
            ydiff = line_smoother_convolve(np.diff(y[:, i]), 11)
            trufts, properties  = find_peaks(-ydiff, height=0, prominence=.005)
            peaks, properties  = find_peaks(ydiff, height=0, prominence=.005)

            npeaks.append(len(peaks))
            ntrufts.append(len(trufts))
        return npeaks, ntrufts

class plotter:
    def __init__(self):
        self.ts = timeseries_returns(load=False, trace=4, do_all_arms=False)
        y, _, _ = self.ts.get_y(self.ts.data)
        ff = features_finder(y)
        self.features = ff.make_df()

        self.features.hist()

    def traces_plotter(self):
        y, _, _ = self.ts.get_y(self.ts.data)
        y = self.ts.array_scaler(y)

        f, axarr = plt.subplots(2, 1)
        axarr[0].set(facecolor=[.2, .2, .2])

        axarr[0].plot(y[:, 0])

        ydiff = line_smoother_convolve(np.diff(y[:, 0]), 11)
        trufts, properties  = find_peaks(-ydiff, height=0, prominence=.005)
        peaks, properties  = find_peaks(ydiff, height=0, prominence=.005)

        
        axarr[1].plot(ydiff)
        axarr[1].plot(trufts, ydiff[trufts], "x")
        axarr[1].plot(peaks, ydiff[peaks], "x")


if __name__ == "__main__":
    p = plotter()
    # p.traces_plotter()


    plt.show()

