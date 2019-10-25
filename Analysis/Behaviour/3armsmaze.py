# %%
import sys
sys.path.append('./')

from Utilities.imports import *
from Analysis.Behaviour.utils.experiments_analyser import ExperimentsAnalyser
%matplotlib inline

# %%
# get data
ea = ExperimentsAnalyser(load_psychometric=False)
ea.conditions = dict(maze0 = ea.load_trials_by_condition(maze_design=0, naive=None, lights=1, escapes=True, escapes_dur=True))
trials = dict(maze0 = ea.load_trials_by_condition(maze_design=0, naive=None, lights=1, escapes=True, escapes_dur=True))['maze0']
# %%
ea.plot_tracking_trace(trials, "tracking_data", as_scatter=False, colorby="arm")


# %%
