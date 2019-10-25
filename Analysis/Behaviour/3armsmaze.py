# %%
import sys
sys.path.append('./')

from Utilities.imports import *
from Analysis.Behaviour.utils.experiments_analyser import ExperimentsAnalyser
from Analysis.Behaviour.utils.T_utils import get_T_data, save_T_tracking_plot
# %matplotlib inline

# %%
# get data
ea = ExperimentsAnalyser(load_psychometric=False)
ea.conditions = dict(maze0 = ea.load_trials_by_condition(maze_design=0, naive=None, lights=1, escapes=True, escapes_dur=True))
trials = dict(maze0 = ea.load_trials_by_condition(maze_design=0, naive=None, lights=1, escapes=True, escapes_dur=True))['maze0']
# %%
ea.plot_tracking_trace(trials, "tracking_data", as_scatter=False, colorby="arm")

# %%
threat_data = get_T_data(median_filter=True, ea=ea)

# %%
save_T_tracking_plot(threat_data, yth=500, ax_kwargs=dict(ylim=[200, 500], xlim=[425, 575]))

# %%
