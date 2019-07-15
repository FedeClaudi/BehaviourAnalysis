import sys
sys.path.append('./')   

from Utilities.imports import *

import pymc3 as pm
from math import factorial as fact
from scipy.special import binom

class Bayes:
    # Bayes hyper params
    hyper_mode = (5, 5)  # a,b of hyper beta distribution (modes)
    concentration_hyper = (0.01, 0.01)  # mean and std of hyper gamma distribution (concentrations)
    

    def __init__(self):
        pass

    def save_bayes_trace(self, trace, savepath):
        if not isinstance(trace, pd.DataFrame):
            trace = pm.trace_to_dataframe(trace)

        with open(savepath, 'wb') as output:
            pickle.dump(trace, output, pickle.HIGHEST_PROTOCOL)

    def load_trace(self, loadname):
        trace = pd.read_pickle(loadname)
        return trace

    def model_hierarchical_bayes(self, conditions):
        hits, ntrials, p_r, n_mice = self.get_binary_trials_per_condition(conditions)

        # Prep hyper and prior params
        k_hyper_shape, k_hyper_rate = gamma_distribution_params(mean=self.concentration_hyper[0], sd=self.concentration_hyper[1])

        # Create model and fit
        n_conditions = len(list(conditions.keys()))
        print("Fitting bayes to conditions:", list(conditions.keys()))
        with pm.Model() as model:
            # Define hyperparams
            modes_hyper = pm.Beta("mode_hyper", alpha=self.hyper_mode[0], beta=self.hyper_mode[1], shape=n_conditions)
            concentrations_hyper = pm.Gamma("concentration_hyper", alpha=k_hyper_shape, beta=k_hyper_rate, shape=n_conditions) # + 2 # ! FIGURE OUT WHAT THIS + 2 IS DOING OVER HERE ????

            # Define priors
            for i, condition in enumerate(conditions.keys()):
                prior_a, prior_b = beta_distribution_params(omega=modes_hyper[i], kappa=concentrations_hyper[i])
                prior = pm.Beta("{}_prior".format(condition), alpha=prior_a, beta=prior_b, shape=len(ntrials[condition]))
                likelihood = pm.Binomial("{}_likelihood".format(condition), n=ntrials[condition], p=prior, observed=hits[condition])

            # Fit
            print("Got all the variables, starting NUTS sampler")
            trace = pm.sample(6000, tune=500, cores=4, nuts_kwargs={'target_accept': 0.99}, progressbar=True)
            

        return trace


    def analytical_bayes_individuals(self, conditions=None, data=None, mode="individuals"):
        """
            Solve bayesian model analytically without the need for MCMC.
            NOT hierarchical model, just individual mice

            # either pass conditions or pass a dataframe with pre loaded data
        """

        if conditions is not None:
            hits, ntrials, p_r, n_mice = self.get_binary_trials_per_condition(conditions)
            # for (cond, H), (c, N) in zip(hits.items(), ntrials.items())
            raise NotImplementedError
        elif data is not None:
            for exp, trials in data.items():
                # Get number of tirals ad hits per session
                N, K = [], []
                sessions = sorted(set(trials.session_uid.values))
                for uid in sessions:
                    sess_trials = trials.loc[trials.session_uid == uid]
                    N.append(len(sess_trials))
                    K.append(len([i for i,t in sess_trials.iterrows() if "right" in t.escape_arm.lower()]))

                if mode == "individuals":
                    # Each mouse outcome is modelled as a binomial given the number of hits k and trials n
                    # The probability of the binomial θ is what we need to estimate given a Beta prior
                    # The priors are Beta distribution with a=b=5 -> mode = .5

                    # Because Beta is a conjugate prior to the Binomial, the posterior is a Beta with
                    # a2 = a1 - 1 + k and b2 = b1 - 1 + n - k and the whole thing is multiplied by
                    # binomial factor n!/(k!(n-k)!)
                    f, ax  = plt.subplots()

                    a, b = 1, 1
                    posteriors = {"a":[], "b":[], "c":[]}
                    for k, n in zip(K, N):
                        a2, b2 = a - 1 + k, b - 1 + n - k
                        posteriors["a"].append(a2)
                        posteriors["b"].append(b2)
                        posteriors["c"].append(binom(n, k))

                        plot_distribution(a2, b2, title=exp, dist_type="beta", xlim=[-0.1, 1.1], ylim=[0, 10], ax=ax)
                    plot_distribution(a, b, dist_type="beta", xlim=[0, 1], ylim=[0, 12], color='g', ax=ax, title=exp)

                elif mode == "grouped":
                    # Compute the likelihood function (product of each mouse's likelihood)and plg that into bayes theorem
                    # the likelihood function will be a Binomial with the binomial factor being the product of all the factors, 
                    # time theta to the product of K times (1-theta) to the product of n-k.
                    f, ax  = plt.subplots()
                    a, b = 100, 100

                    # compute likelihood function
                    fact, kk, dnk = 1, 1, 1
                    for k,n in zip(K, N):
                        fact *= binom(n, k)
                        kk += k
                        dnk += n-k
                    
                    # Now compute the posterior
                    a2 = a - 1 + kk, 
                    b2 = b - 1 + dnk
                    plot_distribution(a2, b2, dist_type="beta", xlim=[0, 1], ylim=[0, 12], ax=ax, title=exp)
                    plot_distribution(a, b, dist_type="beta", xlim=[0, 1], ylim=[0, 12], color='g', ax=ax, title=exp)

                    
                elif mode == "hierarhical":
                    pass
                else: raise ValueError(mode)
                


        else:
            raise ValueError("need to pass either condtions or data")

 