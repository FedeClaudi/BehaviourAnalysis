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
        self.master()

    def master(self):
        # names = ['Left_Far', 'Left_Medium', 'Centre', 'Right_Medium', 'Right_Far', 'all_paths']
        names = ['all_paths']
        f, axarr = plt.subplots(len(names)+1, 1)
        for i, n in enumerate(names):
            self.ts = timeseries_returns(load=False, trace=4, do_all_arms=n)
            self.stats = self.prep_data()
            trials = self.ts.data['is trial'].values
            self.stats['is_trial'] = trials

            self.plotter(axarr[i], n)
            

    def prep_data(self):
        def get_at_shelt(a):
            try:
                at_shelt = np.where(a <= 70)[0][0]
            except:
                return len(a)
            else:
                return at_shelt

        d, v = self.ts.get_y(self.ts.data, sel=4)[0], self.ts.get_y(self.ts.data, sel=2)[0]
        
        at_shelters, max_vels = [], []
        for i in range(d.shape[1]):
            at_shelters.append(get_at_shelt(d[:, i])/30)
            max_vels.append(np.max(v[:, i]))
        
        return pd.DataFrame.from_dict(dict(duration=at_shelters, max_vel=max_vels))
        
    def plotter(self, ax, ttl, ax2=None):        
        for i,c in zip([0, 1], ['r', 'g']):
            sel = self.stats.loc[self.stats['is_trial']==i]
            ax.scatter(sel['duration'].values, sel['max_vel'].values,
                        c=c, s=30, alpha=.4, label=str(i))

            if ax2 is not None:
                ratio = np.divide(sel['duration'].values, sel['max_vel'].values)
                pure = np.linspace(4*i-.5, 4*i+.5, len(ratio))
                noise = np.random.normal(0, .5, pure.shape)
                signal = pure + noise
                ax2.scatter(signal,ratio, s=30, color=c, alpha=.3)
                ax2.set(title='dur/maxv', facecolor=[.2, .2, .2])
        ax.set(title=ttl, xlabel='s', ylabel='px/frame', facecolor=[.2, .2, .2])

if __name__ == "__main__":
    p = clusterer()
    # p.traces_plotter()


    plt.show()

