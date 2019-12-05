"""
    Try to predict each trial's outcome based on the previous trial
"""

# %%
import sys
sys.path.append('./')
from Utilities.imports import *
import statsmodels.api as sm
from pandas.plotting import scatter_matrix

from Analysis.Behaviour.utils.experiments_analyser import ExperimentsAnalyser
# %matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
# %%
# Getting data
ea = ExperimentsAnalyser(load_psychometric=False, tracking="threat")
ea.add_condition("m2", maze_design=2, escapes_dur=True, tracking="threat")
ea.add_condition("m3", maze_design=3, escapes_dur=True, tracking="threat")


trials = ea.merge_conditions_trials(list(ea.conditions.values()))

print("Found {} trials".format(len(trials)))
print("\n", trials.head())
# Explore the data
print("\np(R) : {}".format(round(list(trials.escape_arm).count("right")/len(trials), 2)))



# %%
