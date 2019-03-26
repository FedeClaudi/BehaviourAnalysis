import sys
sys.path.append('./')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple
from itertools import combinations
import time
import scipy.stats as stats
import math
import matplotlib.mlab as mlab
import matplotlib as mpl
from scipy.signal import medfilt as median_filter
from sklearn.preprocessing import normalize
from scipy.integrate import cumtrapz as integral
import seaborn as sns
from tqdm import tqdm

mpl.rcParams['text.color'] = 'k'
mpl.rcParams['xtick.color'] = 'k'
mpl.rcParams['ytick.color'] = 'k'
mpl.rcParams['axes.labelcolor'] = 'k'

from database.NewTablesDefinitions import *

from Processing.tracking_stats.math_utils import *
from Utilities.file_io.files_load_save import load_yaml

from database.database_fetch import *




sessions = sorted(set((AllTrials & "experiment_name='PathInt2'").fetch("session_uid") ))

yth = 400
dth = 20

idphi_l, trackings = [], []
for sess_n, uid in tqdm(enumerate(sessions)):
    trials = (AllTrials & "session_uid={}".format(uid) & "is_escape='true'").fetch("tracking_data")
    # if sess_n > 2: break

    for i, tracking in enumerate(trials):
        
        x, y = median_filter(tracking[:, 0, 1]), median_filter(tracking[:, 1, 1])

        xy = np.vstack([x,y])
        try:
            d = np.cumsum(calc_distance_between_points_in_a_vector_2d(xy.T))
        except:
            continue

        at_yth = np.where(y >= yth)[0][0]
        at_dth = np.where(y > y[0]+dth)[0][0]
        #at_dth = np.where(d >= dth)[0][0]

        x = x[at_dth : at_yth]
        y = y[at_dth : at_yth]

        dx, dy = np.diff(x), np.diff(y)

        # vel = median_filter(np.unwrap(np.arctan2(dy, dx)))
        vel = calc_angle_between_points_of_vector(np.vstack([dx, dy]).T)
        vel[vel==np.nan] = 0

        dv = np.diff(line_smoother(vel)).reshape(len(line_smoother(vel))-1, 1)
        idphi = np.trapz(vel)

        #plt.figure()
        #plt.plot(vel)
        #plt.plot(dv)
        #plt.plot(median_filter(np.rad2deg(np.unwrap(np.arctan2(dy, dx)))))
        #plt.show()


        idphi_l.append(idphi)
        trackings.append(np.vstack([x, y, np.insert(0, 0, vel)]).T)

        # ax.plot(x,y, alpha=.6, label=str(round(idphi[0], 2)))

f, axarr = plt.subplots(ncols = 3)

zdphi = stats.zscore(np.array(idphi_l))

th = 1.25
normal = [t for z,t in zip(zdphi, trackings) if z <= th]
vte = [t for z,t in zip(zdphi, trackings) if z > th]

axarr[0].hist(zdphi, bins=20)

for i, n in enumerate(normal):
    if i > 15: break
    # axarr[1].scatter(n[:, 0], n[:, 1], c=n[:, 2], alpha=.5, vmin=np.min(n[:, 2]), vmax=np.max(n[:, 2]))
    axarr[1].plot(n[:, 0], n[:, 1],alpha=.5)

for i, v in enumerate(vte):
    if i > 25: break
    # axarr[2].scatter(v[:, 0], v[:, 1], c=v[:, 2], alpha=.5)
    axarr[2].plot(v[:, 0], v[:, 1], alpha=.5)

plt.show()