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
    print('Importing choice visualiser')
    from Processing.choice_analysis.chioces_visualiser import ChoicesVisualiser as chioce_data


import matplotlib
# gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
# for gui in gui_env:
#     try:
#         matplotlib.use(gui,warn=False, force=True)
#         from matplotlib import pyplot as plt
#         break
#     except:
#         continue
# print("Using:",matplotlib.get_backend())
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
            np.save('.\\Processing\\modelling\\grouped_traces.npy', burned_trace)

        if display_distributions:
            pm.traceplot(burned_trace)
            # pm.posteriorplot.plot_posterior(burned_trace)

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
        else:
            print('  loading traces') # load stached data and organise them for plotti
            asym_traces = np.load('./Processing/modelling/asym_individual_traces.npy')
            sym_traces = np.load('./Processing/modelling/sym_individual_traces.npy')
            all_traces = np.vstack([asym_traces, sym_traces])
            all_sessions = all_traces.shape[0]
            exp_ids = np.hstack([np.zeros(asym_traces.shape[0]), np.ones(sym_traces.shape[0])])
            individuals = [(np.int(exp_ids[i]), 0, dict(p_individual=all_traces[i, :])) for i in np.arange(all_sessions)]

        
        if display_distributions:
            # [pm.traceplot(burned) for _, _, burned in individuals]

            if not load_traces:
                asym_traces = [burned['p_individual'] for exp, _, burned in individuals if exp==0]
                sym_traces = [burned['p_individual'] for exp, _, burned in individuals if exp==1]

                np.save('.\\Processing\\modelling\\asym_individual_traces.npy', asym_traces)
                np.save('.\\Processing\\modelling\\sym_individual_traces.npy', sym_traces)

            colors=[[.8, .4, .4], [.4, .8, .4]]
            f, axarr = plt.subplots(nrows=4, ncols=2, facecolor=[.2, .2, .2])

            # plot histograms and kde for individual mice
            for exp, trials, burned in individuals:
                axarr[0, exp].hist(burned['p_individual'], bins=100, histtype='step', alpha=.5) # , alpha=trials.mean())
                sns.kdeplot(burned['p_individual'], ax=axarr[1, exp], shade=True, alpha=.7)

            # plot comulative kde
            sns.kdeplot(np.concatenate(asym_traces), ax=axarr[2, 0], color=colors[0], shade=True, alpha=.8, label='individual modelling')  
            sns.kdeplot(np.concatenate(sym_traces), ax=axarr[2, 1], color=colors[1], shade=True, alpha=.8, label='individual modelling')   

            # Plot histograms from grouped modelling
            try:
                # grouped_traces = self.model_grouped(display_distributions=True)
                if sys.platform == 'darwin':
                    grouped_traces_load = np.load('./Processing/modelling/grouped_traces.npy')
                    grouped_traces = {}
                    grouped_traces['p_asym'] = [grouped_traces_load[i]['p_asym'] for i in np.arange(len(grouped_traces_load))]
                    grouped_traces['p_sym'] = [grouped_traces_load[i]['p_sym'] for i in np.arange(len(grouped_traces_load))]
                else: raise NotImplementedError
            except:
                pass
            else:
                # axarr[2, 0].hist(grouped_traces['p_asym'], bins=100, label='grouped modelling', normed=True)
                # axarr[2, 1].hist(grouped_traces['p_sym'], bins=100, label='grouped modelling', normed=True)
                sns.kdeplot(grouped_traces['p_asym'], ax=axarr[3, 0], color=colors[0], label='grouped modelling')
                sns.kdeplot(grouped_traces['p_sym'], ax=axarr[3, 1], color=colors[1], label='grouped modelling')
                axarr[3, 0].set(title="Grouped modelling posterior distribution")
                axarr[3, 1].set(title="Grouped modelling posterior distribution")

            # Add background color and legends to axes
            for ax in axarr.flatten():
                ax.set(xlim=[0, 1], xlabel='p(R)', ylabel='frequency') # facecolor=[.2, .2, .2]
            axarr[0, 0].set(title='ASYM, posterior p(R) individuals')
            axarr[0, 1].set(title='SYM, posterior p(R) individuals')     
            axarr[2, 0].set(title="Comulative posterior KDE")
            axarr[2, 1].set(title="Comulative posterior KDE")

        a = 1
  
    def model_hierarchical(self):
        # Create a dataframe with observed p(R)
        rates_dict = dict(n_trials=[], observed_rates=[], experiment=[], successes=[])
        for session in self.data['session'].unique():
            session_data = self.data.loc[self.data['session']==session]
            rates_dict['n_trials'].append(session_data.shape[0])
            rates_dict['experiment'].append(session_data['experiment'].values[0])
            rates_dict['observed_rates'].append(np.mean(session_data['trial_outcome'].values))
            rates_dict['successes'].append(np.sum(session_data['trial_outcome'].values))

        rates = pd.DataFrame.from_dict(rates_dict)  # <- dataframe of rates

        # Create variables for easier indexing
        asym_n_trials = rates.loc[rates['experiment']==0]['n_trials'].values
        asym_sessions_count =  rates.loc[rates['experiment']==0].shape[0]
        asym_successes = rates.loc[rates['experiment']==0]['successes'].values
        asym_rates = rates.loc[rates['experiment']==0]['observed_rates'].values

        # Model
        with pm.Model() as hierarchical_model:
            """ 
                The p(R) of each mouse is modelled as a Bernoulli distribution with 'n' trials and 'p' probability. 
                    - n is fiexed and is equivalent to the number of trials of each mouse
                    - p is what we want to find out by fitting the model to the observed p(R)

                The true underlying rates are thought to be drawn from a Normal distribution (one for each experiment) with
                mean mu and standard devation std where:
                    - mu has a prior that is uniform between 0 and 1
                    - std is drawn from a half t-test distribution
            """

            # Define hyperpriors: mu and std of Normal distribution
            asym_mu_p = pm.Uniform('asym_mu_p', lower=0,upper=1)
            # sym_mu_p = pm.Uniform('sym_mu_p', lower=0, upper=1)

            asym_sd_p = pm.HalfStudentT('asym_sd_p', nu=3, sd=2.5)
            # sym_sd_p = pm.HalfStudentT('sym_sd_p', nu=3, sd=2.5)

            # Define the two intermediate normal distributions
            asym_p = pm.Normal('asym_p', mu=asym_mu_p, sd=asym_sd_p, shape=asym_sessions_count)
            # sym_p = pm.Normal('sym_p', mu=sym_mu_p, sd=sym_sd_p)

            # Define the individuals binomial distributions
            likelihood = pm.Binomial('likelihood', asym_n_trials, asym_p, observed=asym_successes, shape=asym_sessions_count)
            # estimate = pm.Binomial('estimate', asym_n_trials, asym_p,    shape=asym_sessions_count)

            # likelihood = pm.Bernoulli('likelihoood', p=asym_p, observed=asym_rates, shape=asym_sessions_count)
            # estimate = pm.Bernoulli('estimate', p=asym_p, shape=asym_sessions_count)

            step = pm.Metropolis()
            trace = pm.sample(800000, step=step)
            burned_trace = trace[20000:]

        pm.traceplot(burned_trace)

        # f, axarr = plt.subplots(nrows=2)
        # axarr[0].hist(burned_trace['likelihood'])
        # axarr[1].hist(burned_trace['estimate'])

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

    def summary_plots(self):
        # Load the data
        asym_trials = np.load('Processing/modelling/asym_trials.npy')
        sym_trials = np.load('Processing/modelling/sym_trials.npy')

        grouped_traces_load = np.load('./Processing/modelling/grouped_traces.npy')
        grouped_traces = {}
        grouped_traces['p_asym'] = [grouped_traces_load[i]['p_asym'] for i in np.arange(len(grouped_traces_load))]
        grouped_traces['p_sym'] = [grouped_traces_load[i]['p_sym'] for i in np.arange(len(grouped_traces_load))]


        asym_traces = np.load('./Processing/modelling/asym_individual_traces.npy')
        sym_traces = np.load('./Processing/modelling/sym_individual_traces.npy')
        all_traces = np.vstack([asym_traces, sym_traces])
        all_sessions = all_traces.shape[0]
        exp_ids = np.hstack([np.zeros(asym_traces.shape[0]), np.ones(sym_traces.shape[0])])
        individuals = [(np.int(exp_ids[i]), 0, dict(p_individual=all_traces[i, :])) for i in np.arange(all_sessions)]


        # plt the grouped data
        colors = ["#A60628", "#467821", "#7A68A6"]

        f, ax = plt.subplots()

        sns.kdeplot(grouped_traces['p_asym'], ax=ax, color=colors[0], label='Asymmetric maze', shade=True)
        sns.kdeplot(grouped_traces['p_sym'], ax=ax, color=colors[1], label='Symmetric maze', shade=True)
        ax.axvline(np.nanmean(self.asym_data.all_trials),  color=[.6, .2, .2], linestyle=":")
        ax.axvline(np.nanmean(self.sym_data.all_trials),
                   color=[.2, .6, .2], linestyle=":")

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



if __name__ == "__main__":
    modeller = BayesModeler()
    # modeller.save_data()

    # modeller.model_grouped()
    # modeller.model_individuals()
    # modeller.model_hierarchical()

    modeller.summary_plots()

    plt.show()














