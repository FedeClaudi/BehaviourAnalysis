# %%
import sys
sys.path.append('./')

from Utilities.imports import *
from mpl_toolkits.mplot3d import Axes3D

from Processing.trials_analysis.all_trials_loader import Trials

%matplotlib inline  

# %%
# Get data
trials = Trials(exp_mode=0)

trials = trials.trials.loc[trials.trials["grouped_experiment_name"] == "asymmetric"]



#%%
# Loop over sessions and get speeds distribution during 
sessions = set(sorted(trials.session_uid.values))



f, axarr = plt.subplots(ncols=2)

# all_speeds = 
for uid in sessions:
    # Get speed during the whole session
    speeds = np.vstack((TrackingData.BodyPartData & "bpname='body'" & "uid={}".format(uid)).fetch("tracking_data"))[:, 2]
    axarr[0].hist(speeds, histtype="step", density=True, bins=200, alpha=.5, color="r")

    # Get speed during escapes
    # trials_speeds = np.vstack(trials.loc[trials["session_uid"] == uid].tracking_data.values)[:, 2]
    # axarr[0].hist(trials_speeds, histtype="step", density=True, bins=200, alpha=.5, color="g")
    # axarr[1].hist(trials_speeds, histtype="step", density=True, bins=100, alpha=.5, color="g")

# Plot modelled
shape, rate = gamma_distribution_params(mean=1, sd=1)
modelled = stats.gamma(1.25, loc=0.1, scale=0.5)
x = np.linspace(modelled.ppf(0.00000001), modelled.ppf(0.9999999), 100)
axarr[0].plot(x, modelled.pdf(x), color="g", lw=3)

axarr[0].set(xlim=[-1, 12], ylim=[0, 1.6])
axarr[1].set(xlim=[-1, 12], ylim=[0, 1.6])

#%%
# Plot mean speeds
f, axarr =plt.subplots(ncols=2)
mean_speeds = []
for i , trial in trials.iterrows():
    if "Right" in trial.escape_arm: y = 1
    else: y = -1
    axarr[0].scatter(np.nanmean(trial.tracking_data[:, 2]), y, color="r", alpha=.8)
    mean_speeds.append(np.nanmean(trial.tracking_data[:, 2]))

axarr[1].hist(mean_speeds)

#%%
