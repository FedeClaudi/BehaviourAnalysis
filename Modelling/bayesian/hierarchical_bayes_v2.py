
import sys
sys.path.append('./')

from Utilities.imports import *

import scipy.stats as stats
import pymc3 as pm
from collections import defaultdict
import theano.tensor as tt
from scipy.stats import ks_2samp as KS_test
from scipy import stats
import PyQt5
import pickle

from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame, get_arm_given_rois, convert_roi_id_to_tag

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

            trace = pm.sample(2000, tune=1000, nuts_kwargs={'target_accept': 0.95}) 
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
            # individuals_asym = pm.Uniform("individuals_asym", 0, 1, shape=len(asym_trials))
            # indi_obs_asym = pm.Binomial('indi_obs_asym', n=asym_trials, p=individuals_asym, observed=asym_hits)

            # individuals_sym = pm.Uniform("individuals_sym", 0, 1, shape=len(sym_trials))
            # indi_obs_sym = pm.Binomial('indi_obs_sym', n=sym_trials, p=individuals_sym, observed=sym_hits)

            # Model individuals - hierarchical
            hyperpriors_alfa = pm.Uniform('hyperpriors_alfa', .1, 10, shape=2)
            hyperpriors_beta = pm.Uniform('hyperpriors_beta', .1, 10, shape=2)

            asym_hierarchical = pm.Beta('asym_hierarchical', alpha=hyperpriors_alfa[0], beta=hyperpriors_beta[0], shape=len(asym_trials))
            test = pm.Beta('test', alpha=hyperpriors_alfa[0], beta=hyperpriors_beta[0], shape=len(asym_trials))
            obs_asym_h = pm.Binomial('obs_asym', n=asym_trials, p=asym_hierarchical, observed=asym_hits)

            sym_hierarchical = pm.Beta('sym_hierarchicalsym', alpha=hyperpriors_alfa[1], beta=hyperpriors_beta[1], shape=len(sym_trials))
            test2 = pm.Beta('test2', alpha=hyperpriors_alfa[1], beta=hyperpriors_beta[1], shape=len(asym_trials))
            obs_sym_h = pm.Binomial('obs_sym', n=sym_trials, p=sym_hierarchical, observed=sym_hits)


            # extra params of the beta distributions for visualisation
            kappa_asym = pm.Deterministic('kappa_asym', hyperpriors_alfa[0] + hyperpriors_beta[0])
            kappa_sym = pm.Deterministic('kappa_sym', hyperpriors_alfa[1] + hyperpriors_beta[1])

            mu_asym = pm.Deterministic('mu_asym', hyperpriors_alfa[0] / (hyperpriors_alfa[0] + hyperpriors_beta[0]))
            mu_sym = pm.Deterministic('mu_sym', hyperpriors_alfa[1] / (hyperpriors_alfa[1] + hyperpriors_beta[1]))

            # Model grouped
            asym_grouped = pm.Uniform("p_asym_grouped", 0, 1)
            sym_grouped = pm.Uniform("p_sym_grouped", 0, 1)
            obs_asym = pm.Binomial('obs_asym_grouped', n=asym_trials_grouped, p=asym_grouped, observed=asym_hits_grouped)
            obs_sym = pm.Binomial('obs_sym_grouped', n=sym_trials_grouped, p=sym_grouped, observed=sym_hits_grouped)

            burned_trace = pm.sample(2000, tune=2000, nuts_kwargs={'target_accept': 0.95})

        pm.traceplot(burned_trace,)

        trace = pm.trace_to_dataframe(burned_trace)

        nrows = 6
        f, axarr = plt.subplots(ncols=2, nrows=nrows)
        r = np.random.uniform(0, 10, 10000)
        axarr[0,0].hist(r, histtype='step', color='k', density=True)
        axarr[0,0].hist(trace.hyperpriors_alfa__0, histtype='stepfilled', color='r', alpha=.75, density=True)

        axarr[0,1].hist(r, histtype='step', color='k', density=True)
        axarr[0,1].hist(trace.hyperpriors_alfa__1, histtype='stepfilled', color='r', alpha=.75, density=True)


        axarr[1,0].hist(r, histtype='step', color='k', density=True)
        axarr[1,0].hist(trace.hyperpriors_beta__0, histtype='stepfilled', color='r', alpha=.75, density=True)

        axarr[1,1].hist(r, histtype='step', color='k', density=True)
        axarr[1,1].hist(trace.hyperpriors_beta__1, histtype='stepfilled', color='r', alpha=.75, density=True)

        for c in list(trace.columns):
            if 'asym_hierarchical__' in c:
                ax = axarr[2, 0]
            elif 'sym_hierarchicalsym__' in c:
                ax = axarr[2, 1]
            else: continue
            ax.hist(trace[c], histtype='step', alpha=.75, density=True)

        axarr[3, 0].hist(trace.mu_asym, histtype='stepfilled', color='r', alpha=.75, density=True)
        axarr[3, 1].hist(trace.mu_sym, histtype='stepfilled', color='r', alpha=.75, density=True)
        axarr[4, 0].hist(trace.kappa_asym, histtype='stepfilled', color='r', alpha=.75, density=True)
        axarr[4, 1].hist(trace.kappa_sym, histtype='stepfilled', color='r', alpha=.75, density=True)

        titles = ['A hyper', 'B hyper', 'Posteriors', 'mu', 'kappa']
        for i, t in zip(np.arange(nrows), titles):
            for ii in np.arange(2):
                if i == 0 and ii == 0:
                   t = t + ' asymmetric'
                elif i == 0 and ii == 1:
                    t = t + ' symmetric'

                axarr[i, ii].set(title=t)

        a = pm.model_to_graphviz(model)
        a.render('Processing/modelling/bayesian/test', view=True)
        a.view()

        savename = 'Processing/modelling/bayesian/hb_trace.pkl'
        with open(savename, 'wb') as output:
                    pickle.dump(pm.trace_to_dataframe(burned_trace), output, pickle.HIGHEST_PROTOCOL)

        plt.show()

    def load_trace(self, savename=None):
        if savename is None:
            savename = 'Processing/modelling/bayesian/data/hb_trace.pkl'
        trace = pd.read_pickle(savename)

        # with open(savename, 'rb') as dataload:
        #     trace = pickle.load(dataload)

        return trace

    def save_trace(self, trace, savepath):
        if not isinstance(trace, pd.DataFrame):
            trace = pm.trace_to_dataframe(trace)

        with open(savepath, 'wb') as output:
            pickle.dump(trace, output, pickle.HIGHEST_PROTOCOL)

    def summary_plot(self):
        trace = self.load_trace()

        colors = get_n_colors(10)
        c0, c1 = '#6abf69', '#388e3c'
        f, axarr = plt.subplots(ncols=2, nrows=3, figsize=(16, 10))
        axarr = axarr.flatten()
        cum = False
        lw, a = .75, .05

        f2, ax2 = plt.subplots()
        f3, ax3 = plt.subplots()

        # Plot grouped
        sns.kdeplot(trace['p_asym_grouped'].values, ax=axarr[0], shade=True, color=c0,  linewidth=lw, alpha=.8, cumulative =cum ,clip=[0, 1])
        sns.kdeplot(trace['p_sym_grouped'].values,  ax=axarr[0], shade=True, color=c1, linewidth=lw, alpha=.8, cumulative =cum ,clip=[0, 1])

        # plot individuals shades
        for col in trace.columns:
            if 'individuals_asym' in col:
                sns.kdeplot(trace[col].values, ax=axarr[2], shade=True, color=c0, linewidth=lw, alpha=a, cumulative =cum ,clip=[0, 1])
            elif 'individuals_sym' in col:
                sns.kdeplot(trace[col].values, ax=axarr[3], shade=True, color=c1, linewidth=lw, alpha=a, cumulative =cum ,clip=[0, 1])
            elif 'asym_hierarchical' in col:
                sns.kdeplot(trace[col].values, ax=axarr[4], shade=True, color=c0, linewidth=lw, alpha=a, cumulative =cum ,clip=[0, 1])
            elif 'sym_hierarchicalsym' in col:
                sns.kdeplot(trace[col].values, ax=axarr[5], shade=True, color=c1, linewidth=lw, alpha=a, cumulative =cum ,clip=[0, 1])

        # plot individuals lines
        for col in trace.columns:
            if 'individuals_asym' in col:
                sns.kdeplot(trace[col].values, ax=axarr[2], shade=False, color=c0, linewidth=1, alpha=1, cumulative =cum ,clip=[0, 1])
            elif 'individuals_sym' in col:
                sns.kdeplot(trace[col].values, ax=axarr[3], shade=False, color=c1, linewidth=1, alpha=1, cumulative =cum ,clip=[0, 1])
            elif 'asym_hierarchical' in col:
                sns.kdeplot(trace[col].values, ax=axarr[4], shade=False, color=c0, linewidth=1, alpha=1, cumulative =cum ,clip=[0, 1])
                sns.kdeplot(trace[col].values, ax=ax2, shade=False, color='k', linewidth=1, alpha=1, cumulative =cum ,clip=[0, 1])
            elif 'sym_hierarchicalsym' in col:
                sns.kdeplot(trace[col].values, ax=axarr[5], shade=False, color=c1, linewidth=1, alpha=1, cumulative =cum)
                sns.kdeplot(trace[col].values, ax=ax3, shade=False, color='r', linewidth=1, alpha=1, cumulative =cum ,clip=[0, 1])


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

        ax2.set(xlim=[ 0, 1], ylim=[0, 12])
        ax3.set(xlim=[ 0, 1], ylim=[0, 12])

        f.savefig("Processing/modelling/bayesian/summary_{}.png".format(cum))
        f.savefig("Processing/modelling/bayesian/summary.svg", format="svg")

        # Get delta between grouped sym and grouped asym
        asym_samples = random.choices(list(trace['p_asym_grouped'].values), k=10000)
        sym_samples = random.choices(list(trace['p_sym_grouped'].values), k=10000)

        delta = np.array(asym_samples) - np.array(sym_samples)
        f4, ax4 = plt.subplots()
        sns.kdeplot(delta,  ax=ax4, shade=True, color='k', linewidth=lw, alpha=.8, cumulative =cum ,clip=[0, 1])
        ci = percentile_range(delta)
        ax4.plot([ci.low, ci.high], [10, 10], color='k', linewidth=3)
        ax4.set(title = 'delta asym - sym', xlim=[ 0, 1], ylim=[0, 12])

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

    def better_hierarchical_bayes(self):
        # Prepare data
        asym_escapes, sym_escapes = self.get_individuals_data()
        asym_hits = [np.sum(np.array(trials)) for trials in asym_escapes]
        asym_trials = [len(trials) for trials in asym_escapes]
        sym_hits = [np.sum(np.array(trials)) for trials in sym_escapes]
        sym_trials = [len(trials) for trials in sym_escapes]

        # Set up the model in PYMC3 and run the NUTS sampler to sample the posterior distribution through MCMC
        print("Setting up model")
        mode_hyper_a, mode_hyper_b = 5, 5
        concentration_hyper_mean, concentration_hyper_sd = 1, 10
        concentration_hyper_shape, concentration_hyper_rate = gamma_distribution_params(mean=concentration_hyper_mean, sd=concentration_hyper_sd)

        with pm.Model() as model:
            mode_hyper = pm.Beta("mode_hyper", alpha=mode_hyper_a, beta=mode_hyper_b, shape=2)
            concentration_hyper = pm.Gamma("concentration_hyper", alpha=concentration_hyper_shape, beta=concentration_hyper_rate, shape=2) + 2

            asym_prior_a, asym_prior_b = beta_distribution_params(omega=mode_hyper[0], kappa=concentration_hyper[0])
            asym_prior = pm.Beta("asym_prior", alpha=asym_prior_a, beta=asym_prior_b, shape=len(asym_trials))
            likelihood = pm.Binomial('asym_likelihood', n=asym_trials, p=asym_prior, observed=asym_hits)

            sym_prior_a, sym_prior_b = beta_distribution_params(omega=mode_hyper[1], kappa=concentration_hyper[1])
            sym_prior = pm.Beta("sym_prior", alpha=sym_prior_a, beta=sym_prior_b, shape=len(sym_trials))
            likelihood = pm.Binomial('sym_likelihood', n=sym_trials, p=sym_prior, observed=sym_hits)

            burned_trace = pm.sample(6000, tune=2000, nuts_kwargs={'target_accept': 0.95}, cores=8)

        # Inspect MCMC traces
        pm.traceplot(burned_trace)

        # Make other plots
        trace = pm.trace_to_dataframe(burned_trace)
        f, axarr = plt.subplots(2, 2)

        _ho = np.random.beta(mode_hyper_a,  mode_hyper_b, 10000)   
        _hk = np.random.gamma(concentration_hyper_shape, concentration_hyper_rate, 10000)     
        
        axarr[0, 0].hist(_ho, color='k', histtype='step', alpha=.75, density=True)
        axarr[0, 0].hist(trace.mode_hyper__0, color='r', histtype='stepfilled', alpha=.75, density=True)
        axarr[0, 0].hist(trace.mode_hyper__1, color='g', histtype='stepfilled', alpha=.75, density=True)

        axarr[0, 0].set(title="hyper. mode")

        axarr[0, 1].hist(_hk, color='k', histtype='step', alpha=.75, density=True)
        axarr[0, 1].hist(trace.concentration_hyper__0, color='r', histtype='stepfilled', alpha=.75, density=True)
        axarr[0, 1].set(title="hyper. concentration")


        a,b = beta_distribution_params(omega=np.max(trace.mode_hyper__0.values), kappa=np.max(trace.concentration_hyper__0.values))
        asym_pop = np.random.beta(a, b, 10000)

        a,b = beta_distribution_params(omega=np.max(trace.mode_hyper__1.values), kappa=np.max(trace.concentration_hyper__1.values))
        sym_pop = np.random.beta(a, b, 10000)

        _a, _b = beta_distribution_params(omega=.5, kappa=5)
        _pop = np.random.beta(_a, _b, 10000)

        axarr[1, 0].hist(_pop, color='k', histtype='step', alpha=.75, density=True)
        axarr[1, 0].hist(asym_pop, color='r', histtype='stepfilled', alpha=.75, density=True)
        axarr[1, 0].hist(sym_pop, color='g', histtype='stepfilled', alpha=.75, density=True)
        axarr[1, 0].set(title="Population")

        for c in list(trace.columns):
            if "prior__" in c:
                axarr[1, 1].hist(trace[c].values,  histtype='step', alpha=.75, density=True)



        # plot_two_dists_kde(None, trace.mode_hyper__0.values, trace.mode_hyper__1.values, 'Pop Mode')
        # plot_two_dists_kde(None, trace.concentration_hyper__0.values, trace.concentration_hyper__1.values, 'Pop Concentration', no_ax_lim=True)

        self.save_trace(trace, "Processing\\modelling\\bayesian\\hierarchical_v2_2.pkl")
        plt.show()

if __name__ == "__main__":
    mod = Modeller()
    mod.better_hierarchical_bayes()
    # mod.summary_plot()
