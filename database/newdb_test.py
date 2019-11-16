import sys

sys.path.append("./")
from Utilities.imports import *
from Analysis.Behaviour.utils.experiments_analyser import ExperimentsAnalyser

ea = ExperimentsAnalyser(load_psychometric=True, escapes_dur=True)
ea.add_condition("m6", maze_design=6, escapes_dur=True)

a = ea.plot_pr_bayes_bycond()

# ea.arm_preference_plot_bycond()
