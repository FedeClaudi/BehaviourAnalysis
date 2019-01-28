import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

mpl.rcParams['text.color'] = 'k'
mpl.rcParams['xtick.color'] = 'k'
mpl.rcParams['ytick.color'] = 'k'
mpl.rcParams['axes.labelcolor'] = 'k'



import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy
import scipy.stats as stats

from Processing.choice_analysis.chioces_visualiser import ChoicesVisualiser as chioce_data


class BayesModeler:
    def __init__(self):
        data = chioce_data(run=False)

        # Get all binary outcomes for both experiments
        datatuple = namedtuple('data', 'all_trials by_session')
        asym_binary, asym_binary_by_session = data.get_experiment_binary_outcomes('PathInt2')
        sym_binary, sym_binary_by_session = data.get_experiment_binary_outcomes('Square Maze')

        self.asym_data = datatuple(asym_binary, asym_binary_by_session) # All trials grouped together and trials grouped by session
        self.sym_data = datatuple(sym_binary, sym_binary_by_session)

        self.print_data_summary()
        self.data = self.organise_data_in_df()

        # Initialise empty variables, to be filled with modeling
        self.grouped_samples = None
        self.individuals_samples = None

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
            n_trials = d.by_session.shape[1]

            print("""
            {}:
                - Group mean p(R):      {}
                - Mean of p(R):         {} 
                - Number of sessions:   {}
                - Number of trials:     {}
            
            """.format(n, mean_of_individual_probs, grouped_prob, n_sessions, n_trials, ))


    def model_grouped(self, display_distributions=True):
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
            trace = pm.sample(20000 , step=step)
            burned_trace=trace[5000:]

        self.grouped_samples = dict(asym = burned_trace["p_asym"], sym = burned_trace["p_sym"], delta=burned_trace['delta'])

        if display_distributions:
            pm.traceplot(burned_trace)

            f, axarr = plt.subplots(nrows=4, facecolor=[.2, .2, .2], figsize=(10, 10))

            priors_samples = (p_asym.random(size=20000),  p_sym.random(size=20000))
            priors = ['p_asym', 'p_sym']
            for samp, prior in zip(priors_samples, priors):
                axarr[0].hist(samp, bins=100, alpha=.5, normed=True, histtype="stepfilled", label=prior)
            axarr[0].set(title='Priors', facecolor=[.2, .2, .2], xlim=[-.1, 1.1])
            axarr[0].legend()

            axarr[1].hist(burned_trace["p_asym"], bins=100, label='Asym posterior')
            axarr[1].hist(burned_trace['p_sym'], bins=100, label='Sym posterior')
            axarr[1].set(title='Posteriors', facecolor=[.2, .2, .2], xlim=[-.1, 1.1])
            axarr[1].legend


            axarr[2].hist(self.asym_data.all_trials, color='w', normed=True,  label='Asym Trials')
            axarr[2].hist(burned_trace['predicted_A'], color='r', alpha=.5,  normed=True, label='Est. Asym Trials')
            axarr[2].set(title='asym trials', xlim=[-.5, 1.5], facecolor=[.2, .2, .2])
            axarr[2].legend()


            axarr[3].hist(self.sym_data.all_trials, color='w', normed=True, label='Sym Trials')
            axarr[3].hist(burned_trace['predicted_B'], color='r', alpha=.5,  normed=True, label='Est. Sym Trials')
            axarr[3].set(title='sym trials', xlim=[-.5, 1.5], facecolor=[.2, .2, .2])
            axarr[3].legend()





    def model_individuals(self, display_distributions=True):
        print('\n\nClustering individuals')
        sessions = set(self.data['session'])
        
        individuals = []
        for session in sessions:
            # if session == 10: break

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
        
        if display_distributions:
            [pm.traceplot(burned) for _, _, burned in individuals]

        asym_traces = [burned['p_individual'] for exp, _, burned in individuals if exp==0]

        colors=[[.8, .4, .4], [.4, .8, .4]]
        f, axarr = plt.subplots(nrows=2, ncols=2, facecolor=[.2, .2, .2])
        for exp, trials, burned in individuals:
            axarr[0, exp].hist(burned['p_individual'], bins=100, histtype='step', alpha=.5) # , alpha=trials.mean())
            sns.kdeplot(burned['p_individual'], ax=axarr[1, exp], shade=True, alpha=.1)
        for ax in axarr.flatten():
            ax.set(facecolor=[.2, .2, .2], xlim=[0, 1], xlabel='p(R)', ylabel='frequency')
        axarr[0, 0].set(title='ASYM, posterior p(R) individuals')
        axarr[0, 1].set(title='SYM, posterior p(R) individuals')
                


        # self.individuals_samples = [(i, p, t) for i,p,t in zip(dataset_record, p_individuals, n_trials_individuals)]
        # a=1


    def model_hierarchical(self):
        asym_trials = self.data.loc[self.data['experiment']==0]['trial_outcome']
        asym_sessions = len(set(asym_trials['session']))


        with pm.Model() as hierarchical_model:
            # Hyper prior
            mu_p = pm.Normal('mu_p', mu=.5, alpha=1)
            sigma_p = pm.HalfCauchy('sigma_p', beta=1)

            # p for each mouse which is drawn from a Normal with mean mu_p
            p = np.Normal('p', mu=mu_p, sd=sigma_p, shape=asym_sessions)

            # likelihood and estimated
            hierarchical_p = pm.Bernoulli('hieararchical_p')


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



if __name__ == "__main__":
    modeller = BayesModeler()
    # modeller.save_data()

    # modeller.model_grouped()
    modeller.model_individuals()
    
    # modeller.plot_posteriors_histo()

    plt.show()














