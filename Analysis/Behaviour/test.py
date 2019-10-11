# %%
import sys
sys.path.append('./')

from Utilities.imports import *
from Analysis.Behaviour.experiments_analyser import ExperimentsAnalyser
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import find_peaks



# %%
ea = ExperimentsAnalyser(load=False,  naive=None, lights=1, escapes=False, escapes_dur=False, shelter=False)
# Get Threat Platform data
# ea.prep_tplatf_trials_data(filt=False, remove_errors=True, speed_th=None)
%matplotlib inline

#%%
ea.plot_trials_tracking()

#%%
trials = ea.get_sessions_trials()

#%%
f, ax = create_figure(subplots=False, figsize=(20, 20))


for i, trial in trials.iterrows():
    ax.plot( trial.tracking_data[:, 0],  trial.tracking_data[:, 1], alpha=1, lw=1) 

ax.set(facecolor=[.2, .2, .2])
#%%
