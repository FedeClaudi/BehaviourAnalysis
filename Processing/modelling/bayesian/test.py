import PyQt5

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

import numpy as np
import scipy.stats as stats
import pymc3 as pm
import pandas as pd
from collections import defaultdict
import theano.tensor as tt
import yaml

import sys
sys.path.append('./')
if sys.platform != 'darwin':
    from database.NewTablesDefinitions import *
    from database.database_fetch import *

from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame, get_arm_given_rois, convert_roi_id_to_tag
from Processing.tracking_stats.math_utils import get_roi_enters_exits, line_smoother, calc_distance_between_points_2d, remove_tracking_errors


class Modeller:
    def __init__(self):
        self.platform = sys.platform

    def get_grouped_data(self):
        # Get data [L-R escapes]
        asym_exps = ["PathInt2", "PathInt2-L"]
        sym_exps = ["Square Maze", "TwoAndahalf Maze"]

        asym = [arm for arms in [get_trials_by_exp(e, 'true', ['escape_arm']) for e in asym_exps] for arm in arms]
        sym = [arm for arms in [get_trials_by_exp(e, 'true', ['escape_arm']) for e in sym_exps] for arm in arms]

        asym_escapes = np.array([1 if 'Right' in e else 0 for e in asym])
        sym_escapes = np.array([1 if 'Right' in e else 0 for e in sym])

        np.save('Processing/modelling/bayesian/asym.npy', asym_escapes)
        np.save('Processing/modelling/bayesian/sym.npy', sym_escapes)

        return asym_escapes, sym_escapes

    def get_individuals_data(self):
        asym_exps = ["PathInt2", "PathInt2-L"]
        sym_exps = ["Square Maze", "TwoAndahalf Maze"]

        asym_escapes = []
        for exp in asym_exps:
            sessions = get_trials_by_exp(exp, 'true', ['session_uid'])
            for uid in set(sessions):
                asym_escapes.append(get_trials_by_exp_and_session(exp, uid, 'true', ['escape_arm']))

        sym_escapes = []
        for exp in sym_exps:
            sessions = get_trials_by_exp(exp, 'true', ['session_uid'])
            for uid in set(sessions):
                sym_escapes.append(get_trials_by_exp_and_session(exp, uid, 'true', ['escape_arm']))

        with open('Processing/modelling/bayesian/asym_individuals.yml', 'w') as out:
            yaml.dump(out, asym_escapes)
        with open('Processing/modelling/bayesian/asym_individuals.yml', 'w') as out:
            yaml.dump(out, sym_escapes)

    def model_grouped(self):
        if self.platform == 'darwin':
            asym_escapes = np.load('Processing/modelling/bayesian/asym.npy')
            sym_escapes = np.load('Processing/modelling/bayesian/sym.npy')
        else:
            asym_escapes, sym_escapes = self.get_grouped_data()

        print("Building model")
        with pm.Model() as model:
            p_asym = pm.Uniform("p_asym", 0, 1)
            p_sym = pm.Uniform("p_sym", 0, 1)

            obs_asym = pm.Bernoulli("obs_asym", p_asym, observed=asym_escapes)
            obs_sym = pm.Bernoulli("obs_sym", p_sym, observed=sym_escapes)

            # step = pm.Metropolis()
            trace = pm.sample(6000) # , step=step)
            burned_trace = trace[1000:]

        pm.traceplot(burned_trace)
        pm.posteriorplot.plot_posterior(burned_trace)
        plt.show()



if __name__ == "__main__":
    mod = Modeller()
    mod.get_individuals_data()