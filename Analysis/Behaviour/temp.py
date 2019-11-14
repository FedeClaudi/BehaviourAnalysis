# %%
import sys
sys.path.append('./')

from Utilities.imports import *
from Analysis.Behaviour.utils.experiments_analyser import ExperimentsAnalyser
# %matplotlib inline

# %%
ea = ExperimentsAnalyser(naive=None, lights=1, escapes=False, escapes_dur=True,  shelter=1, load_psychometric=False,
                    agent_params={'kernel_size':3, 'model_path':'PathInt2_old.png', 'grid_size':1000})

ea.conditions = dict(
                        maze1 =  ea.load_trials_by_condition(maze_design=1),
                        maze2 =  ea.load_trials_by_condition(maze_design=2),
                        maze3 =  ea.load_trials_by_condition(maze_design=3),
                        maze4 =  ea.load_trials_by_condition(maze_design=4),
                        m6 = ea.load_trials_by_condition(maze_design=6))

ea.pr_by_condition()
plt.show()

# %%
