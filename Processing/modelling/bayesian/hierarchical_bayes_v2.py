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
import pickle 
import seaborn as sns

from scipy.stats import ks_2samp as KS_test
from scipy import stats

import sys
sys.path.append('./')
if sys.platform != 'darwin':
    from database.NewTablesDefinitions import *
    from database.database_fetch import *

from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame, get_arm_given_rois, convert_roi_id_to_tag
from Processing.tracking_stats.math_utils import get_n_colors, get_roi_enters_exits, line_smoother, calc_distance_between_points_2d, remove_tracking_errors


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

    def model_two_distributions(self, d1, d2):
        d1_hits, d1_count = np.sum(d1), len(d1)
        d2_hits, d2_count = np.sum(d2), len(d2)

        with pm.Model() as model:
            p_d1 = pm.Uniform("p_d1", 0, 1)
            p_d2 = pm.Uniform("p_d2", 0, 1)

            obs_asym = pm.Binomial('obs_asym', n=d1_count, p=p_d1, observed=d1_hits)
            obs_sym = pm.Binomial('obs_sym', n=d2_count, p=p_d2, observed=d2_hits)

            trace = pm.sample(3000, tune=1000, nuts_kwargs={'target_accept': 0.95}) 
        # pm.traceplot(trace)
        df = pm.trace_to_dataframe(trace)

        D, ks_pVal, t, t_pval = self.stats(distributions=[df['p_d1'].values, df['p_d2'].values])

        # print('KS test:', KS_stat, pVal)
        return df, D, ks_pVal, t, t_pval

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
            # model individuals - not hierarcical
            individuals_asym = pm.Uniform("individuals_asym", 0, 1, shape=len(asym_trials))
            indi_obs_asym = pm.Binomial('indi_obs_asym', n=asym_trials, p=individuals_asym, observed=asym_hits)

            individuals_sym = pm.Uniform("individuals_sym", 0, 1, shape=len(sym_trials))
            indi_obs_sym = pm.Binomial('indi_obs_sym', n=sym_trials, p=individuals_sym, observed=sym_hits)

            # Model individuals - hierarchical
            hyperpriors_alfa = pm.Uniform('hyperpriors_alfa', .1, 10, shape=2)
            hyperpriors_beta = pm.Uniform('hyperpriors_beta', .1, 10, shape=2)

            asym_hierarchical = pm.Beta('asym_hierarchical', alpha=hyperpriors_alfa[0], beta=hyperpriors_beta[0], shape=len(asym_trials))
            obs_asym = pm.Binomial('obs_asym', n=asym_trials, p=asym_hierarchical, observed=asym_hits)
            

            sym_hierarchical = pm.Beta('sym_hierarchicalsym', alpha=hyperpriors_alfa[1], beta=hyperpriors_beta[1], shape=len(sym_trials))
            obs_sym = pm.Binomial('obs_sym', n=sym_trials, p=sym_hierarchical, observed=sym_hits)

            # Model grouped
            asym_grouped = pm.Uniform("p_asym_grouped", 0, 1)
            obs_asym = pm.Binomial('obs_asym_grouped', n=asym_trials_grouped, p=asym_grouped, observed=asym_hits_grouped)

            sym_grouped = pm.Uniform("p_sym_grouped", 0, 1)
            obs_sym = pm.Binomial('obs_sym_grouped', n=sym_trials_grouped, p=sym_grouped, observed=sym_hits_grouped)



            burned_trace = pm.sample(1000, tune=2000, nuts_kwargs={'target_accept': 0.95}) # , cores=1, chains=4)

        pm.traceplot(burned_trace,)

        a = pm.model_to_graphviz(model)
        a.render('Processing/modelling/bayesian/test', view=True)
        a.view()

        savename = 'Processing/modelling/bayesian/hb_trace.pkl'
        with open(savename, 'wb') as output:
                    pickle.dump(pm.trace_to_dataframe(burned_trace), output, pickle.HIGHEST_PROTOCOL)

        plt.show()

    def load_trace(self):
        savename = 'Processing/modelling/bayesian/hb_trace.pkl'
        with open(savename, 'rb') as dataload:
            trace = pickle.load(dataload)
        return trace

    def summary_plot(self):
        trace = self.load_trace()

        colors = get_n_colors(6)
        f, axarr = plt.subplots(ncols=2, nrows=3, figsize=(16, 10))
        axarr = axarr.flatten()
        cum = False
        f2, ax2 = plt.subplots()
        # Plot grouped
        sns.kdeplot(trace['p_asym_grouped'].values, ax=axarr[0], shade=True, color=colors[2],  linewidth=2, alpha=.8, cumulative =cum ,clip=[0, 1])
        sns.kdeplot(trace['p_sym_grouped'].values,  ax=axarr[0], shade=True, color=colors[-1], linewidth=2, alpha=.8, cumulative =cum ,clip=[0, 1])

        # plot individuals shades
        for col in trace.columns:
            if 'individuals_asym' in col:
                sns.kdeplot(trace[col].values, ax=axarr[2], shade=True, color=colors[2], linewidth=2, alpha=.05, cumulative =cum ,clip=[0, 1])
            elif 'individuals_sym' in col:
                sns.kdeplot(trace[col].values, ax=axarr[3], shade=True, color=colors[-1], linewidth=2, alpha=.05, cumulative =cum ,clip=[0, 1])
            elif 'asym_hierarchical' in col:
                sns.kdeplot(trace[col].values, ax=axarr[4], shade=True, color=colors[2], linewidth=2, alpha=.05, cumulative =cum ,clip=[0, 1])
            elif 'sym_hierarchicalsym' in col:
                sns.kdeplot(trace[col].values, ax=axarr[5], shade=True, color=colors[-1], linewidth=2, alpha=.05, cumulative =cum ,clip=[0, 1])

        # plot individuals lines
        for col in trace.columns:
            if 'individuals_asym' in col:
                sns.kdeplot(trace[col].values, ax=axarr[2], shade=False, color=colors[2], linewidth=2, alpha=1, cumulative =cum ,clip=[0, 1])
            elif 'individuals_sym' in col:
                sns.kdeplot(trace[col].values, ax=axarr[3], shade=False, color=colors[-1], linewidth=2, alpha=1, cumulative =cum ,clip=[0, 1])
            elif 'asym_hierarchical' in col:
                sns.kdeplot(trace[col].values, ax=axarr[4], shade=False, color=colors[2], linewidth=2, alpha=1, cumulative =cum ,clip=[0, 1])
                sns.kdeplot(trace[col].values, ax=ax2, shade=False, color=colors[2], linewidth=2, alpha=1, cumulative =cum ,clip=[0, 1])
            elif 'sym_hierarchicalsym' in col:
                sns.kdeplot(trace[col].values, ax=axarr[5], shade=False, color=colors[-1], linewidth=2, alpha=1, cumulative =cum)
                sns.kdeplot(trace[col].values, ax=ax2, shade=False, color=colors[-1], linewidth=2, alpha=1, cumulative =cum ,clip=[0, 1])


        titles = [
            'Asymmetric - grouped',
            'Symmetric - grouped',
            'Asymmetric - individuals',
            'Symmetric - individuals',
            'Asymmetric - hierarchical',
            'Symmetric - hierarchical'
        ]

        for ttl, ax in zip(titles, axarr):
            ax.set(title=ttl, xlim=[0, 1], xlabel='p(R)', ylabel='frequency')

        f.savefig("Processing/modelling/bayesian/summary_{}.png".format(cum))
        f.savefig("Processing/modelling/bayesian/summary.svg", format="svg")

        plt.show()


    def stats(self, distributions=None):
        if distributions is None:
            trace = self.load_trace()
            d1 = trace['p_asym_grouped'].values
            d2 = trace['p_sym_grouped'].values
        else:
            d1, d2 = distributions[0], distributions[1]

        d, p1 = KS_test(d1, d2)
        t, p2 =  stats.ttest_ind(d1, d2)

        return d, p1, t, p2



if __name__ == "__main__":
    mod = Modeller()
    mod.stats()
