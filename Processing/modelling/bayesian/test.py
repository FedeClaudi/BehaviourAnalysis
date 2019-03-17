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
        if self.platform != "darwin":
            # Get data [L-R escapes]
            asym_exps = ["PathInt2", "PathInt2-L"]
            sym_exps = ["Square Maze", "TwoAndahalf Maze"]

            asym = [arm for arms in [get_trials_by_exp(e, 'true', ['escape_arm']) for e in asym_exps] for arm in arms]
            sym = [arm for arms in [get_trials_by_exp(e, 'true', ['escape_arm']) for e in sym_exps] for arm in arms]

            asym_escapes = np.array([1 if 'Right' in e else 0 for e in asym])
            sym_escapes = np.array([1 if 'Right' in e else 0 for e in sym])

            np.save('Processing/modelling/bayesian/asym.npy', asym_escapes)
            np.save('Processing/modelling/bayesian/sym.npy', sym_escapes)
        else:
            asym_escapes = np.load('Processing/modelling/bayesian/asym.npy')
            sym_escapes = np.load('Processing/modelling/bayesian/sym.npy')

        return asym_escapes, sym_escapes

    def get_individuals_data(self):
        if self.platform != "darwin":
            asym_exps = ["PathInt2", "PathInt2-L"]
            sym_exps = ["Square Maze", "TwoAndahalf Maze"]

            asym_escapes = []
            for exp in asym_exps:
                sessions = get_trials_by_exp(exp, 'true', ['session_uid'])
                for uid in set(sessions):
                    arms = get_trials_by_exp_and_session(exp, uid, 'true', ['escape_arm'])
                    asym_escapes.append([1 if 'Right' in a else 0 for a in arms])

            sym_escapes = []
            for exp in sym_exps:
                sessions = get_trials_by_exp(exp, 'true', ['session_uid'])
                for uid in set(sessions):
                    arms = get_trials_by_exp_and_session(exp, uid, 'true', ['escape_arm'])
                    sym_escapes.append([1 if 'Right' in a else 0 for a in arms])

            with open('Processing/modelling/bayesian/asym_individuals.yml', 'w') as out:
                yaml.dump(asym_escapes, out)
            with open('Processing/modelling/bayesian/sym_individuals.yml', 'w') as out:
                yaml.dump(sym_escapes, out)
        else:
            with open("Processing/modelling/bayesian/asym_individuals.yml", 'r') as inp:
                asym_escapes = yaml.load(inp)

            with open("Processing/modelling/bayesian/sym_individuals.yml", 'r') as inp:
                sym_escapes = yaml.load(inp)

        return asym_escapes, sym_escapes

    def model_grouped(self):
        if self.platform == 'darwin':
            asym_escapes = np.load('Processing/modelling/bayesian/asym.npy')
            sym_escapes = np.load('Processing/modelling/bayesian/sym.npy')
        else:
            asym_escapes, sym_escapes = self.get_grouped_data()

        asym_hits, asym_trials = np.sum(asym_escapes), len(asym_escapes)
        sym_hits, sym_trials = np.sum(sym_escapes), len(sym_escapes)

        print("Building model")
        with pm.Model() as model:
            p_asym = pm.Uniform("p_asym", 0, 1)
            p_sym = pm.Uniform("p_sym", 0, 1)

            # obs_asym = pm.Bernoulli("obs_asym", p_asym, observed=asym_escapes)
            # obs_sym = pm.Bernoulli("obs_sym", p_sym, observed=sym_escapes)

            obs_asym = pm.Binomial('obs_asym', n=asym_trials, p=p_asym, observed=asym_hits)
            obs_sym = pm.Binomial('obs_sym', n=sym_trials, p=p_sym, observed=sym_hits)

            # step = pm.Metropolis()
            trace = pm.sample(6000) # , step=step)
            burned_trace = trace[1000:]

        pm.traceplot(burned_trace)
        pm.posteriorplot.plot_posterior(burned_trace)
        plt.show()


    def model_individuals(self):

        asym_escapes, sym_escapes = self.get_individuals_data()
        # print(np.array(asym_escapes))
        # asym_escapes, sym_escapes = self.get_grouped_data()

        asym_hits = [np.sum(np.array(trials)) for trials in asym_escapes]
        asym_trials = [len(trials) for trials in asym_escapes]
        sym_hits = [np.sum(np.array(trials)) for trials in sym_escapes]
        sym_trials = [len(trials) for trials in sym_escapes]


        print("Setting up model")
        with pm.Model() as model:
            p_asym = pm.Uniform("p_asym", 0, 1, shape=len(asym_trials))
            obs_asym = pm.Binomial('obs_asym', n=asym_trials, p=p_asym, observed=asym_hits)

            p_sym = pm.Uniform("p_sym", 0, 1, shape=len(sym_trials))
            obs_sym = pm.Binomial('obs_sym', n=sym_trials, p=p_sym, observed=sym_hits)

            burned_trace = pm.sample(3000, tune=1000, nuts_kwargs={'target_accept': 0.95})

        pm.traceplot(burned_trace)
        # pm.posteriorplot.plot_posterior(burned_trace)

        plt.show()

    def model_individuals_hierarchical(self):

        asym_escapes, sym_escapes = self.get_individuals_data()
        
        asym_hits = [np.sum(np.array(trials)) for trials in asym_escapes]
        asym_trials = [len(trials) for trials in asym_escapes]
        sym_hits = [np.sum(np.array(trials)) for trials in sym_escapes]
        sym_trials = [len(trials) for trials in sym_escapes]

        asym_escapes_grouped, sym_escapes_grouped = self.get_grouped_data()
        asym_hits_grouped, asym_trials_grouped = np.sum(asym_escapes_grouped), len(asym_escapes_grouped)
        sym_hits_grouped, sym_trials_grouped = np.sum(sym_escapes_grouped), len(sym_escapes_grouped)


        print("Setting up model")
        with pm.Model() as model:
            # Model individuals
            alpha = pm.Uniform('alpha', 1, 10, shape=2)
            beta = pm.Uniform('beta', 1, 10, shape=2)

            asym = pm.Beta('asym', alpha=alpha[0], beta=beta[0], shape=len(asym_trials))
            obs_asym = pm.Binomial('obs_asym', n=asym_trials, p=asym, observed=asym_hits)
            

            sym = pm.Beta('sym', alpha=alpha[1], beta=beta[1], shape=len(sym_trials))
            obs_sym = pm.Binomial('obs_sym', n=sym_trials, p=sym, observed=sym_hits)

            # Model grouped
            p_asym = pm.Uniform("p_asym_grouped", 0, 1)
            obs_asym = pm.Binomial('obs_asym_grouped', n=asym_trials_grouped, p=p_asym, observed=asym_hits_grouped)

            p_sym = pm.Uniform("p_sym_grouped", 0, 1)
            obs_sym = pm.Binomial('obs_sym_grouped', n=sym_trials_grouped, p=p_sym, observed=sym_hits_grouped)



            burned_trace = pm.sample(1000, tune=2000, nuts_kwargs={'target_accept': 0.95}) # , cores=1, chains=4)

        pm.traceplot(burned_trace,)
        # pm.posteriorplot.plot_posterior(burned_trace)

        plt.show()


if __name__ == "__main__":
    mod = Modeller()
    mod.model_individuals_hierarchical()
