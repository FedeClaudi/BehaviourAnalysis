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

import numpy as np
import scipy.stats as stats
import itertools
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy
import scipy.stats as stats

if sys.platform != 'darwin':
    print('Importing choice visualiser')
    from Processing.choice_analysis.chioces_visualiser import ChoicesVisualiser as chioce_data


import matplotlib
gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
for gui in gui_env:
    try:
        matplotlib.use(gui,warn=False, force=True)
        from matplotlib import pyplot as plt
        break
    except:
        continue
print("Using:",matplotlib.get_backend())
from matplotlib import pyplot as plt
# import matplotlib.patches as patches
# import matplotlib.mlab as mlab
# import matplotlib as mpl

# mpl.rcParams['text.color'] = 'w'
# mpl.rcParams['xtick.color'] = 'w'
# mpl.rcParams['ytick.color'] = 'w'
# mpl.rcParams['axes.labelcolor'] = 'w'

class BayesModeler:
    # TODO Convert by_session numpy array to pandas df with a row for each trial (mouse id, exp, trial_number ...)
    # to avoid having nans in the dataset


    def __init__(self, load_data=False):
        if sys.platform == "darwin": load_data = True
        
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
            asym_by_session = np.load('Processing/modelling/asym_trials.npy')
            sym_by_session = np.load('Processing/modelling/sym_trials.npy')
            
            self.asym_data = datatuple(asym_by_session.flatten(), asym_by_session)
            self.sym_data = datatuple(sym_by_session.flatten(), sym_by_session)

        self.print_data_summary()

        # Initialise empty variables, to be filled with modeling
        self.grouped_samples = None
        self.individuals_samples = None


        # ? Clean up data
        self.clean_up_data()

    def clean_up_data(self):
        # data =   self.asym_data.by_session
        # # data = data[~np.isnan(data)]
        # print(np.nan_to_num(data))
        pass



    def save_data(self):
        try:
            np.save('.\\Processing\\modelling\\asym_trials.npy', self.asym_data.by_session)
            np.save('.\\Processing\\modelling\\sym_trials.npy', self.sym_data.by_session)
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

    def model_grouped(self):
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
            # p_asym = pm.Uniform("p_asym", 0, 1)
            # p_sym = pm.Uniform("p_sym", 0, 1)
            
            p_asym = pm.Normal("p_asym", 0, 1)
            p_sym = pm.Normal("p_sym", 0, 1)
            

            # Define the deterministic delta function. This is our unknown of interest.
            delta = pm.Deterministic("delta", p_asym - p_sym)


            # Set of observations, in this case we have two observation datasets.
            obs_asym = pm.Bernoulli("obs_A", p_asym, observed=self.asym_data.all_trials)
            obs_sym = pm.Bernoulli("obs_B", p_sym, observed=self.sym_data.all_trials)

            # Fit the model 
            step = pm.Metropolis()
            trace = pm.sample(20000, step=step)
            burned_trace=trace[1000:]

        self.grouped_samples = dict(asym = burned_trace["p_asym"], sym = burned_trace["p_sym"], delta=burned_trace['delta'])

    def model_individuals(self):
        print('\n\nClustering individuals')
        # Create a record of which sessions belong to which dataset
        asym, sym = self.asym_data.by_session, self.sym_data.by_session
        n_sessions_asym = asym.shape[0]
        n_sessions_sym = sym.shape[0]
        tot_sessions = n_sessions_asym + n_sessions_sym

        dataset_record = np.array([0 if i < n_sessions_asym else 1 for i in np.arange(tot_sessions)])

        p_individuals, n_trials_individuals = [], []
        for session_n, session_dataset in enumerate(dataset_record):
            print('\n\n Modelling session {} of {}'.format(session_n+1, len(dataset_record)))
            # if session_n > 2: continue
            if session_dataset == 0:
                data = asym[session_n, :]
            else:
                data = sym[session_n-asym.shape[0], :]
            data = data[~np.isnan(data)] # ? remove nans
            n_trials = len(data)
            # if n_trials == 0:
            #     a = 1
            #     continue

            with pm.Model() as individual_model:
                p_individual = pm.Uniform("p_individual", 0, 1)
                # obs_individual = pm.Bernoulli("obs_individual", p_individual, observed=self.asym_data.all_trials)

                # Fit the model 
                step = pm.Metropolis()
                trace = pm.sample(20000, step=step)
                burned_trace=trace[1000:]

                p_individuals.append(burned_trace['p_individual'])
                n_trials_individuals.append(n_trials)

        self.individuals_samples = [(i, p, t) for i,p,t in zip(dataset_record, p_individuals, n_trials_individuals)]
        a=1

    def model_hierarchical(self):
        asym = self.asym_data.by_session
        n_sessions = asym.shape[0]
        n_trials = np.tile(asym.shape[0], n_sessions)
        observed_rates = np.nanmean(asym, 1)
        observed_counts = observed_counts = np.nansum(asym, 1)

        with pm.Model() as model:
            mu, sigma = np.nanmean(observed_rates), np.nanstd(observed_rates)

            true_p = pm.Normal('true_p', mu=mu, sd=sigma, shape=n_sessions)
            observed_values = pm.Binomial('observed_values', n_trials, true_p, observed=observed_counts)

            # Fit the model 
            step = pm.Metropolis()
            trace = pm.sample(20000, step=step)
            burned_trace=trace[1000:]

        f, ax = plt.subplots()
        [ax.hist(burned_trace['true_p'][:][:, i], alpha=.2) for i in range(n_sessions)]
        ax.set(xlim=[0, 1])
        a = 1

    def plot_posteriors_histo(self):

        colors = ["#A60628", "#467821", "#7A68A6"]

        if self.grouped_samples is not None:

            #histogram of posteriors
            f, axarr = plt.subplots(nrows=2, facecolor=[.2, .2, .2])

            for (name, traces), color in zip(self.grouped_samples.items(), colors):
                if name == 'delta': continue
                axarr[0].set(facecolor=[.2, .2, .2], xlim=[0, 1], title='Posteriors of GROUPED modelling')
                axarr[0].hist(traces, histtype='stepfilled', bins=25, alpha=0.85,
                        label="posterior of ${}$".format(name), color=color, normed=True)
                # ax.vlines(np.mean(asym_binary), 0, 20, linestyle="--", label="true $p_A$ (unknown)", color='w')
                axarr[0].legend(loc="upper left")

            axarr[1].set(facecolor=[.2, .2, .2])
            axarr[1].hist(self.grouped_samples['delta'], histtype='stepfilled', bins=25, alpha=0.85,
                    label="posterior of $delta$", color=colors[-1], normed=True)
            # ax.vlines(np.mean(asym_binary), 0, 20, linestyle="--", label="true $p_A$ (unknown)", color='w')
            axarr[1].legend(loc="upper right")


        if self.individuals_samples is not None:
            #histogram of posteriors
            f, axarr = plt.subplots(nrows=len(self.individuals_samples), facecolor=[.2, .2, .2])
            datasets = ['Asym', 'Sym']
            for ax, (dataset, traces, n_trials) in zip(axarr, self.individuals_samples):
                color=colors[dataset]

                ax.set(facecolor=[.2, .2, .2], xlim=[0, 1], title='Posteriors of GROUPED modelling - {} - #{} trials'.format(datasets[dataset], n_trials))
                ax.hist(traces, histtype='stepfilled', bins=25, alpha=0.85,
                        color=color, normed=True)

                ax.legend(loc="upper left")



def tests():
    #set constants
    p_true = 0.05  # remember, this is unknown.
    N = 15000

    # sample N Bernoulli random variables from Ber(0.05).
    # each random variable has a 0.05 chance of being a 1.
    # this is the data-generation step
    occurrences = stats.bernoulli.rvs(p_true, size=N)

    with pm.Model() as model:
        p = pm.Uniform('p', lower=0, upper=1)
        obs = pm.Bernoulli("obs", p, observed=occurrences)
        step = pm.Metropolis()
        trace = pm.sample(18000, step=step)
        burned_trace = trace[1000:]

    print(burned_trace['p'].mean(), burned_trace['p'].std())
    a = 1

if __name__ == "__main__":
    # modeller = BayesModeler()
    # # modeller.save_data()
    # # modeller.model_hierarchical()
    # modeller.model_grouped()
    
    # modeller.plot_posteriors_histo()

    tests()

    # plt.show()














