# %%
import sys
sys.path.append('./')

from Utilities.imports import *
from Analysis.Behaviour.experiments_analyser import ExperimentsAnalyser
%matplotlib inline

# %%
# get data
ea = ExperimentsAnalyser(load_psychometric=False)
ea.conditions = dict(maze0 = ea.get_sessions_trials(maze_design=0, naive=None, lights=1, escapes=True, escapes_dur=True))
trials = dict(maze0 = ea.get_sessions_trials(maze_design=0, naive=None, lights=1, escapes=True, escapes_dur=True))
# %%
