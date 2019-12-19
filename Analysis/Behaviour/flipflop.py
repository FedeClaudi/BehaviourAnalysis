# %%
import sys
sys.path.append('./')
from Utilities.imports import *

from Analysis.Behaviour.utils.experiments_analyser import ExperimentsAnalyser
palette = makePalette(teal, lilla, 4, False)
arms_colors = {
    "left": teal,
    'center': darkcyan,
    "right": lilla,
}



# %%
# Getting data
ea = ExperimentsAnalyser(load_psychometric=False, tracking="all")
ea.max_duration_th = 9
ea.add_condition("ff", maze_design=None, lights=None, escapes_dur=True, tracking="all", experiment_name="FlipFlop Maze"); print("Got FlipFlop Maze")
ea.add_condition("ff2", maze_design=None, lights=None, escapes_dur=True, tracking="all", experiment_name="FlipFlop2 Maze"); print("Got FlipFlop2 Maze")

for condition, trials in ea.conditions.items():
    print("Maze {} -- {} trials".format(condition, len(trials)))

# plot tracking
f, axarr = create_figure(subplots=True, ncols=int(np.ceil(len(ea.conditions.keys())/2)), nrows=2)
for n, (condition, trials) in enumerate(ea.conditions.items()):
    for i, trial in trials.iterrows():
        axarr[n].plot(trial.body_xy[:, 0], trial.body_xy[:, 1], color=arms_colors[trial.escape_arm])
    axarr[n].set(title=condition, xlim=[0, 1000], ylim=[0, 1000])


trials = pd.concat([ea.conditions['ff'], ea.conditions['ff2']])

# %%
# TODO GET all the stimuli and trials for each mouse in the FF experiments, and compare with my notes
# TODO then make metadata to know the state of the maze at each trial, split the data and get p(R) grouped and individuals
# TODO then find a way to extract the maze state from the video and look at the behaviour immediately after the flip and compare with baseline exploration


# %%
# ------------------------ GET STIMULI FOR EACH MOUSE ------------------------ #
stims = pd.DataFrame((Stimuli * Session & "experiment_name='FlipFlop2 Maze'").fetch())
sessions = set(stims.uid.values)

for sess in sessions:
    trials = stims.loc[stims.uid == sess]
    data = trials.date.values[0]