import sys
sys.path.append('./')

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from collections import namedtuple
from itertools import combinations
import time
import scipy.stats as stats
import math
import matplotlib.mlab as mlab
import matplotlib as mpl
import seaborn as sns
import os

mpl.rcParams['text.color'] = 'k'
mpl.rcParams['xtick.color'] = 'k'
mpl.rcParams['ytick.color'] = 'k'
mpl.rcParams['axes.labelcolor'] = 'k'

import numpy as np
import scipy.stats as stats
import itertools
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy
import scipy.stats as stats

if sys.platform != 'darwin':
    from Processing.choice_analysis.chioces_visualiser import ChoicesVisualiser as chioce_data

import matplotlib
from matplotlib import pyplot as plt


class BayesModeler:
    def __init__(self,  load_data=False):
        if sys.platform == "darwin": 
            load_data = True
            self.data_fld = './Processing/modelling/data'
        else:
            self.data_fld = '.\\Processing\\modelling\\data'
        cleanup_data = True

        # ? Get Data
        datatuple = namedtuple('data', 'all_trials by_session')
        if not load_data:
            # Get data from database
            data = chioce_data(run=False)

            # Get all binary outcomes for both experiments
            
            asym_binary, asym_binary_by_session = data.get_experiment_binary_outcomes('PathInt2')
            sym_binary, sym_binary_by_session = data.get_experiment_binary_outcomes('Square Maze')

            self.asym_data = datatuple(asym_binary, asym_binary_by_session) # All trials grouped together and trials grouped by session
            self.sym_data = datatuple(sym_binary, sym_binary_by_session)

        else:
            # Load previously saved data
            try: 
                asym_by_session = np.load('Processing/modelling/asym_trials.npy')
                sym_by_session = np.load('Processing/modelling/sym_trials.npy')
                
                self.asym_data = datatuple(asym_by_session.flatten(), asym_by_session)
                self.sym_data = datatuple(sym_by_session.flatten(), sym_by_session)
            except: 
                cleanup_data = False

        if cleanup_data:
            self.print_data_summary()
            self.data = self.organise_data_in_df()


    def organise_data_in_df(self):
        asym_trials = self.asym_data.by_session
        sym_trials = self.sym_data.by_session

        asym_sessions, sym_session = asym_trials.shape[0], sym_trials.shape[0]

        data = dict(session = [], trial_n=[], trial_outcome=[], experiment=[])
        for session_n in np.arange(asym_sessions + sym_session):
            if session_n < asym_sessions:
                trials = asym_trials[session_n, :]
                exp = 0
            else:
                trials = sym_trials[session_n-asym_sessions, :]
                exp = 1

            trials = trials[~np.isnan(trials)]
            for trial_n, trial in enumerate(trials):
                data['session'].append(session_n)
                data['trial_n'].append(trial_n)
                data['trial_outcome'].append(np.int(trial))
                data['experiment'].append(exp)

        datadf = pd.DataFrame.from_dict(data)
        
        return datadf

    def save_data(self):
        try:
            np.save(os.path.join(self.data_fld, 'asym_trials.npy'), self.asym_data.by_session)
            np.save(os.path.join('sym_trials.npy'), self.sym_data.by_session)
        except:
            print('Did not save')
        
    def print_data_summary(self):
        data = [self.asym_data, self.sym_data]
        names = ['Asym data', 'Sym data']

        for d,n in zip(data, names):
            grouped_prob = np.round(np.mean(d.all_trials),2)
            individuals_probs = np.round(np.nanmean(d.by_session, 1), 2)
            n_sessions = d.by_session.shape[0]
            mean_of_individual_probs = np.round(np.nanmean(individuals_probs), 2)
            n_trials = np.sum(d.by_session)

            print("""
            {}:
                - Group mean p(R):      {}
                - Mean of p(R):         {} 
                - Number of sessions:   {}
                - Number of trials:     {}
            
            """.format(n, mean_of_individual_probs, grouped_prob, n_sessions, n_trials, ))

    ################################################################################
    ################################################################################
    ################################################################################

    def model_grouped(self, display_distributions=True, save_traces=True):
        print('\n\nClustering groups')
        """
            Assume that the sequence of L/R trials for each mouse is generated through a Bernoulli 
            distribution with same 'p' for all mice. So we group all trials together and we estimate
            a single value of p for all mice. [The fact that we have different # trials between the
            two conditions is not a problem in Bayesian statistic. It will be reflected in the width
            of the posterior distribution]

            To use an objective prior we assume that for both experiment the prior on 'p' is a uniform
            distribution in [0, 1].
        """

        # Set up the pymc3 model.  assume Uniform priors for p_asym and p_sym.
        with pm.Model() as model:
            p_asym = pm.Uniform("p_asym", 0, 1)
            p_sym = pm.Uniform("p_sym", 0, 1)

            # bounded_normal = pm.Bound(pm.Normal, lower=0, upper=1)
            # p_asym = bounded_normal("p_asym", mu=0.1, sd=.1)
            # p_sym = bounded_normal("p_sym", mu=0.9, sd=.1)


            # Define the deterministic delta function. This is our unknown of interest.
            delta = pm.Deterministic("delta", p_asym - p_sym)

            # Set of observations, in this case we have two observation datasets.
            obs_asym = pm.Bernoulli("obs_A", p_asym, observed=self.asym_data.all_trials)
            obs_sym = pm.Bernoulli("obs_B", p_sym, observed=self.sym_data.all_trials)

            # Estimated posterior distributions to use for 
            est_asym =  pm.Bernoulli("predicted_A", p_asym)
            est_sym =  pm.Bernoulli("predicted_B", p_sym)

            # Fit the model 
            step = pm.Metropolis()
            trace = pm.sample(18000 , step=step)
            burned_trace=trace[1000:]

        self.grouped_samples = dict(asym = burned_trace["p_asym"], sym = burned_trace["p_sym"], delta=burned_trace['delta'])

        if save_traces:
            np.save(os.path.join(self.data_fld, 'grouped_traces.npy'), burned_trace)

        if display_distributions:
            pm.traceplot(burned_trace)
            pm.posteriorplot.plot_posterior(burned_trace)
        return burned_trace

    def model_individuals(self, display_distributions=True, load_traces=True):
        print('\n\nClustering individuals')
        
        if sys.platform == "darwin":
            load_traces = True

        if not load_traces:
            individuals = []
            sessions = set(self.data['session'])
            for session in sessions:
                # if session == 3: break
                print('Processing session {} of {}'.format(session, self.data['session'].values[-1]))
                trials_df = self.data.loc[self.data['session'] == session]
                exp_id = trials_df['experiment'].values[0]
                trials = trials_df['trial_outcome'].values

                with pm.Model() as individual_model:
                    p_individual = pm.Uniform("p_individual", 0, 1)
                    obs_individual = pm.Bernoulli("obs_individual", p_individual, observed=trials)
                    est_individual = pm.Bernoulli("est_individual", p_individual)

                    # Fit the model
                    step = pm.Metropolis()
                    trace = pm.sample(10000, step=step)
                    burned_trace=trace[1000:]

                individuals.append((exp_id, trials, burned_trace))

            # Savee data
            asym_traces = [burned['p_individual'] for exp, _, burned in individuals if exp==0]
            sym_traces = [burned['p_individual'] for exp, _, burned in individuals if exp==1]

            np.save(os.path.join(self.data, 'asym_individual_traces.npy'), asym_traces)
            np.save(os.path.join(self.data, 'sym_individual_traces.npy'), sym_traces)

        else:
            print('  loading traces') # load stached data and organise them for plotti
            asym_traces = np.load(os.path.join(self.data, 'asym_individual_traces.npy'))
            sym_traces = np.load(os.path.join(self.data, 'sym_individual_traces.npy'))
            all_traces = np.vstack([asym_traces, sym_traces])
            all_sessions = all_traces.shape[0]
            exp_ids = np.hstack([np.zeros(asym_traces.shape[0]), np.ones(sym_traces.shape[0])])
            individuals = [(np.int(exp_ids[i]), 0, dict(p_individual=all_traces[i, :])) for i in np.arange(all_sessions)]
        
        if display_distributions:
            [pm.traceplot(burned) for _, _, burned in individuals]


    def model_hierarchical(self, save_traces=True):
        """
            The p(R) of each mouse is modelled as a Bernoulli distribution with 'n' trials and 'p' probability.
                - n is fiexed and is equivalent to the number of trials of each mouse
                - p is what we want to find out by fitting the model to the observed p(R)

            The true underlying rates are thought to be drawn from a Normal distribution (one for each experiment) with
            mean mu and standard devation std where:
                - mu has a prior that is uniform between 0 and 1
                - std is drawn from a half t-test distribution
        """

        if save_traces:
            print('Hierarchical modelling... ')
            # Ge the observed p(R) for each mouse in each experiment
            rates = []
            for session in self.data['session'].unique():
                session_data = self.data.loc[self.data['session'] == session]
                exp = session_data['experiment'].values[0]
                rate = np.mean(session_data['trial_outcome'].values)
                trials = session_data['trial_outcome'].values
                rates.append((exp, rate, trials))

            # Rearrange data
            asym_rates = [r for e,r,t in rates if e == 0]
            asym_n_sessions = len(asym_rates)
            asym_index = np.arange(asym_n_sessions)
            asym_ftrials = stats.bernoulli.rvs(p=np.array(asym_rates), size=(15, asym_n_sessions)) # ! Fake trials, to create an array with uniforme size

            sym_rates = [r for e,r,t in rates if e == 1]
            sym_n_sessions = len(sym_rates)
            sym_index = np.arange(sym_n_sessions)
            sym_ftrials = stats.bernoulli.rvs(p=np.array(sym_rates), size=(15, sym_n_sessions)) # ! Fake trials, to create an array with uniforme size

            asym_trials = [t for e,r,t in rates if e == 0]
            n_trials = [len(t) for t in asym_trials]
            max_t = np.max(n_trials)+1
            empty = [np.full(max_t, np.nan) for t in np.arange(len(asym_trials))]
            asym_padded = []
            for i, (t,n,e) in enumerate(zip(asym_trials, n_trials, empty)):
                e[:n] = t
                asym_padded.append(e)
            
            sym_trials = [t for e,r,t in rates if e == 1]
            n_trials = [len(t) for t in sym_trials]
            max_t = np.max(n_trials)+1
            empty = [np.full(max_t, np.nan) for t in np.arange(len(sym_trials))]
            sym_padded = []
            for i, (t,n,e) in enumerate(zip(sym_trials, n_trials, empty)):
                e[:n] = t
                sym_padded.append(e)

            fps = [os.path.join(self.data_fld, 'part_pooled_asym'), 
                    os.path.join(self.data_fld, 'part_pooled_sym')]
            # datasets = [(asym_n_sessions, asym_index, asym_ftrials, fps[0]),
            #             (sym_n_sessions, sym_index, sym_ftrials, fps[1])]
            datasets = [(asym_n_sessions, asym_index, np.vstack(asym_padded).T, fps[0]), (sym_n_sessions, sym_index, np.vstack(sym_padded).T, fps[1])]

            traces = []
            for n_sessions, index, ftrials, path in datasets:
                print('Ready to model')
                
                # Create PyMC3 model
                with pm.Model() as model:
                    """ 
                        We want to create a hierarchical model in which each individual's trials are assumed
                        to be generated through a bernoulli distribution, we want to discover
                        the true underlying rate of each individual's distribution. 
                        To do so we assume that each individual's rate is pulled from a normal distribution, to
                        model the fact that we think that all individuals have a similar strategy.
                    """

                    # Priors
                    mu_a = pm.Normal('mu_a', mu=0.5, sd=0.1) # 1e5)
                    sigma_a = pm.HalfCauchy('sigma_a', 5)

                    # Random intercepts
                    a = pm.Normal('a', mu=mu_a, sd=sigma_a) # , shape=n_sessions)

                    # Model error
                    sigma_y = pm.HalfCauchy('sigma_y', 5)

                    # Expected value
                    # y_hat = a[index]

                    # Data likelihood
                    y_like = pm.Normal('y_like', mu=a, sd=sigma_y, observed=ftrials) # mu_y_hat

                    step = pm.Metropolis()
                    partial_pooling_trace = pm.sample(5000, tune=1000, njobs=1)
                    # partial_pooling_trace = pm.sample(20000, step=step, tune=2000)

                # pm.traceplot(partial_pooling_trace)
                print(partial_pooling_trace['a'].shape)
                for i in np.arange(partial_pooling_trace['a'].shape[1]):
                    np.save(path+'_{}.npy'.format(i), partial_pooling_trace['a'][:, i])

        else:
            traces = [np.load(os.path.join(self.data, 'part_pooled_asym.npy')), 
                        np.load(os.path.join(self.data, 'part_pooled_sym.npy'))]

    ################################################################################
    ################################################################################
    ################################################################################

    def summary_plots(self):
        # Load the data
        asym_trials = np.load(os.path.join(self.data, 'asym_trials.npy'))
        sym_trials = np.load(os.path.join(self.data, 'sym_trials.npy'))

        grouped_traces_load = np.load('./Processing/modelling/grouped_traces.npy')
        grouped_traces = {}
        grouped_traces['p_asym'] = [grouped_traces_load[i]['p_asym'] for i in np.arange(len(grouped_traces_load))]
        grouped_traces['p_sym'] = [grouped_traces_load[i]['p_sym'] for i in np.arange(len(grouped_traces_load))]


        asym_traces = np.load(os.path.join(self.data, 'asym_individual_traces.npy'))
        sym_traces = np.load(os.path.join(self.data, 'sym_individual_traces.npy'))
        all_traces = np.vstack([asym_traces, sym_traces])
        all_sessions = all_traces.shape[0]
        exp_ids = np.hstack([np.zeros(asym_traces.shape[0]), np.ones(sym_traces.shape[0])])
        individuals = [(np.int(exp_ids[i]), 0, dict(p_individual=all_traces[i, :])) for i in np.arange(all_sessions)]


        pooled_files = [os.path.join(self.data_fld, f) for f in os.listdir(self.data_fld) if 'part_pooled' in f]
        asym_traces = [np.load(f) for f in pooled_files if 'asym' in f]
        asym_traces = np.vstack(asym_traces)
        sym_traces = [np.load(f) for f in pooled_files if not 'asym' in f]
        sym_traces = np.vstack(sym_traces)
        traces = [asym_traces, sym_traces]

        # plt the grouped data
        colors = ["#A60628", "#467821", "#7A68A6"]

        f, ax = plt.subplots()

        sns.kdeplot(grouped_traces['p_asym'], ax=ax, color=colors[0], label='Asymmetric maze', shade=True)
        sns.kdeplot(grouped_traces['p_sym'], ax=ax, color=colors[1], label='Symmetric maze', shade=True)
        ax.axvline(np.nanmean(self.asym_data.all_trials),  color=[.6, .2, .2], linestyle=":")
        ax.axvline(np.nanmean(self.sym_data.all_trials), color=[.2, .6, .2], linestyle=":")

        ax.legend()
        ax.set(title='Grouped model poseterior $p(R)$', xlabel='$p(R)$', ylabel='density', xlim=[0, 1])

        # Plot the individuals data
        f, axarr = plt.subplots(nrows=4)

        for exp, trials, burned in individuals:
            sns.kdeplot(burned['p_individual'], ax=axarr[exp], color=colors[exp], shade=True, alpha=.05, linewidth=2)
            sns.kdeplot(burned['p_individual'], ax=axarr[exp], color=colors[exp], shade=False, linewidth=1.5, alpha=.8)

        sns.kdeplot(np.concatenate(asym_traces), ax=axarr[2], color=colors[0], shade=True, alpha=.3, )  
        sns.kdeplot(np.concatenate(sym_traces), ax=axarr[2], color=colors[1], shade=True, alpha=.3, )   
        sns.kdeplot(np.concatenate(asym_traces), ax=axarr[2], color=colors[0], shade=False, alpha=.8, label='Asymmetric maze')  
        sns.kdeplot(np.concatenate(sym_traces), ax=axarr[2], color=colors[1], shade=False, alpha=.8, label='Symmetric maze')   

        sns.kdeplot(grouped_traces['p_asym'], ax=axarr[-1], color=colors[0], label='Asymmetric maze', shade=True)
        sns.kdeplot(grouped_traces['p_sym'], ax=axarr[-1], color=colors[1], label='Symmetric maze', shade=True)

        axarr[0].set(title='Asym. maze - individuals p(R) posterior', xlabel='$p(R)$', ylabel='Density', xlim=[0, 1])
        axarr[1].set(title='Sym. maze - individuals p(R) posterior', xlabel='$p(R)$', ylabel='Density', xlim=[0, 1])
        axarr[2].set(title='Comulative p(R) posterior', xlabel='$p(R)$', ylabel='Density', xlim=[0, 1])
        axarr[3].set(title='Grouped p(R) posterior', xlabel='$p(R)$', ylabel='Density', xlim=[0, 1])

        axarr[2].legend()
        axarr[3].legend()

        # Plot the partial pooled model
        f, axarr = plt.subplots(nrows=3)

        for i in [0, 1]:
            print(traces[0].shape, traces[1].shape)
            for trace in np.arange(traces[i].shape[0]):
                    sns.kdeplot(traces[i][trace, :], ax=axarr[i], color=colors[i], shade=True, alpha=.1)
                    sns.kdeplot(traces[i][trace, :], ax=axarr[i], color=colors[i], shade=False, alpha=.8)

        sns.kdeplot(np.concatenate(traces[0]), ax=axarr[2], color=colors[0], shade=True, alpha=.1)
        sns.kdeplot(np.concatenate(traces[0]), ax=axarr[2], color=colors[0], shade=False, alpha=.8)
        sns.kdeplot(np.concatenate(traces[1]), ax=axarr[2], color=colors[1], shade=True, alpha=.1)
        sns.kdeplot(np.concatenate(traces[1]), ax=axarr[2], color=colors[1], shade=False, alpha=.8)

        axarr[0].set(title='Partial pooled model - Asym. posteriors', xlabel='$p(R)$', ylabel='density', xlim=[0, 1])
        axarr[1].set(title='Partial pooled model - Sym. posteriors', xlabel='$p(R)$', ylabel='density', xlim=[0, 1])
        axarr[2].set(title='Comulative of posteriors', xlabel='$p(R)$', ylabel='density', xlim=[0, 1])



if __name__ == "__main__":
    modeller = BayesModeler()
    # modeller.save_data()

    # modeller.model_grouped()
    # modeller.model_individuals()
    modeller.model_hierarchical()

    # modeller.summary_plots()

    plt.show()














