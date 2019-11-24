# %%
import sys
sys.path.append('./')
from Utilities.imports import *
import statsmodels.api as sm
from pandas.plotting import scatter_matrix

from Analysis.Behaviour.utils.experiments_analyser import ExperimentsAnalyser

# %%
# Getting data
ea = ExperimentsAnalyser(load_psychometric=False, tracking="all")
ea.add_condition("m1", maze_design=1, escapes_dur=True, tracking="all")
ea.add_condition("m2", maze_design=2, escapes_dur=True, tracking="all")
ea.add_condition("m3", maze_design=3, escapes_dur=True, tracking="all")
ea.add_condition("m4", maze_design=4, escapes_dur=True, tracking="all")
ea.add_condition("m6", maze_design=6, escapes_dur=True, tracking="all")


# %%
def plot_euclidean_distance(trials):
    f, ax = plt.subplots(figsize=(5, 5))
    for i, trial in trials.iterrows():
        start_frame = trial.out_of_t_frame - trial.stim_frame
        tracking = trial.body_xy[start_frame:, :]
        ax.scatter(tracking[:, 0], tracking[:, 1], c=trial.shelter_distance, vmin=0, vmax=400, 
                    cmap="Reds", s=5, alpha=.8)


# %%
# Compute shelter distance
shelter_location = [500, 650]
for condition, trials in ea.conditions.items():
    shelter_distance = []
    for i, trial in trials.iterrows():
        start_frame = trial.out_of_t_frame - trial.stim_frame
        tracking = trial.body_xy[start_frame:, :]
        shelter_distance.append(calc_distance_from_shelter(tracking, shelter_location))
    trials['shelter_distance'] = shelter_distance
    
    plot_euclidean_distance(trials)



# %%

# %%
