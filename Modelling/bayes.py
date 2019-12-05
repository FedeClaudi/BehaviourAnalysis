import sys
sys.path.append('./')   

from Utilities.imports import *

import pymc3 as pm
from math import factorial as fact
from scipy.special import binom
import pickle
# import pymc3 as pm
import pydot

class Bayes:
    # Bayes hyper params
    hyper_mode = (1, 1)  # a,b of hyper beta distribution (modes)
    concentration_hyper = (1, 10)  # mean and std of hyper gamma distribution (concentrations)
    k_hyper_shape, k_hyper_rate = 0.01, 0.01

    a, b  = 1.00, 1.00 # Parameters of priors Beta for individual and grouped analytical solution

    def __init__(self):
        pass

    """
        DATA IO UTILS
    """

    def save_bayes_trace(self, trace, savepath):
        if not isinstance(trace, pd.DataFrame):
            trace = pm.trace_to_dataframe(trace)

        with open(savepath, 'wb') as output:
            pickle.dump(trace, output, pickle.HIGHEST_PROTOCOL)

    def load_trace(self, loadname):
        trace = pd.read_pickle(loadname)
        return trace
    
    def save_model_image(self, model, savepath):
        model = pm.model_to_graphviz(model)
        model.render(filename=savepath)


    """
        ANALYSIS
    """

    def grouped_bayes_analytical(self, n, k):
        """[Solves the bayesia model for p(R) for grouped data]
        
        Arguments:
            n {[int]} -- [tot number of trials]
            k {[int]} -- [tot number of hits]
        """
        # Compute posterior function
        fact = binom(n, k)
        a2 = self.a + k - 1
        b2 = self.b + n -k -1

        # Plot mean and mode of posterior
        mean =  a2 / (a2 + b2)
        mode = (a2 -1)/(a2 + b2 -2)
        sigmasquared = (a2 * b2)/((a2+b2)**2 * (a2 + b2 + 1)) # https://math.stackexchange.com/questions/497577/mean-and-variance-of-beta-distributions
        prange = percentile_range(get_parametric_distribution("beta", a2, b2)[1])

        return (a2, b2, mean, mode, sigmasquared, prange)
     
        

    def model_hierarchical_bayes(self, conditions): # ? using PyMC3
        hits, ntrials, p_r, n_mice, _ = self.get_binary_trials_per_condition(conditions)

        # Create model and fit
        n_conditions = len(list(conditions.keys()))
        print("Fitting bayes to conditions:", list(conditions.keys()))
        with pm.Model() as model:
            # Define hyperparams
            modes_hyper = pm.Beta("mode_hyper", alpha=self.hyper_mode[0], beta=self.hyper_mode[1], shape=n_conditions)
            concentrations_hyper = pm.Gamma("concentration_hyper", alpha=self.k_hyper_shape, beta=self.k_hyper_rate, shape=n_conditions)

            # Define priors
            for i, condition in enumerate(conditions.keys()):
                prior_a, prior_b = beta_distribution_params(omega=modes_hyper[i], kappa=concentrations_hyper[i])
                prior = pm.Beta("{}_prior".format(condition), alpha=prior_a, beta=prior_b, shape=len(ntrials[condition]))
                likelihood = pm.Binomial("{}_likelihood".format(condition), n=ntrials[condition], p=prior, observed=hits[condition])

            # Fit
            print("Got all the variables, starting NUTS sampler")
            self.save_model_image(model, os.path.join("D:\\Dropbox (UCL - SWC)\\Rotation_vte\\plots\\00_Psychometric", "hb"))
            # trace = pm.sample(6000, tune=500, cores=2, nuts_kwargs={'target_accept': 0.99}, progressbar=True)

            # pm.traceplot(trace)
            plt.show()
        return trace

    def analytical_bayes_individuals(self, conditions=None, data=None, mode="individuals", plot=True):
        """
            Solve bayesian model analytically without the need for MCMC.
            NOT hierarchical model, just individual mice

            # either pass conditions or pass a dataframe with pre loaded data
        """
        a, b  = self.a, self.b
        ptuple = namedtuple("betaparams", "a b")
        
        if conditions is not None:
            hits, ntrials, p_r, n_mice, _ = self.get_binary_trials_per_condition(conditions)
            # for (cond, H), (c, N) in zip(hits.items(), ntrials.items())
            
            # compute likelihood function
            modes, means, params, sigmas, pranges = {}, {}, {},{}, {}
            for condition in conditions.keys():
                fact, kk, dnk = 1, 1, 1
                for k,n in zip(hits[condition], ntrials[condition]):
                    fact *= binom(n, k)
                    kk += k
                    dnk += n-k
            
                # Now compute the posterior
                a2 = a - 1 + kk
                b2 = b - 1 + dnk

                # Plot mean and mode of posterior
                mean =  a2 / (a2 + b2)
                _mode = (a2 -1)/(a2 + b2 -2)
                sigmasquared = (a2 * b2)/((a2+b2)**2 * (a2 + b2 + 1)) # https://math.stackexchange.com/questions/497577/mean-and-variance-of-beta-distributions
                prange = percentile_range(get_parametric_distribution("beta", a2, b2)[1])
                
                modes[condition], means[condition], params[condition], sigmas[condition], pranges[condition] = _mode, mean, ptuple(a2, b2), sigmasquared, prange
            return modes, means, params, sigmas, pranges


        elif data is not None:
            if plot: f, ax  = plt.subplots(figsize=(12, 12),)
            
            modes, means, params, sigmas, pranges = {}, {}, {},{}, {}
            for expn, (exp, trials) in enumerate(data.items()):
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
                    # for k, n in zip(K, N):
                    #     a2, b2 = a - 1 + k, b - 1 + n - k
                    #     if plot: 
                    #         plot_distribution(a2, b2, dist_type="beta", ax=ax,
                    #                                 plot_kwargs=dict(color=self.colors[expn+1], 
                    #                                 ax_kwargs=dict(xlim=[-0.1, 1.1], 
                    #                                                 ylim=[0, 10]), )
                    pass
                    # if plot:
                    #     if expn == 0:
                    #         plot_distribution(a, b,dist_type="beta", ax=ax,
                    #                                 plot_kwargs=dict(color="w", 
                    #                                                 label="prior"), 
                    #                                 ax_kwargs=dict(xlim=[-0.1, 1.1], 
                    #                                                 ylim=[0, 15],
                    #                                                 title=exp))
                elif mode == "grouped":
                    # Compute the likelihood function (product of each mouse's likelihood)and plg that into bayes theorem
                    # the likelihood function will be a Binomial with the binomial factor being the product of all the factors, 
                    # time theta to the product of K times (1-theta) to the product of n-k.

                    # compute likelihood function
                    fact, kk, dnk = 1, 1, 1
                    for k,n in zip(K, N):
                        fact *= binom(n, k)
                        kk += k
                        dnk += n-k
                    
                    # Now compute the posterior
                    a2 = a - 1 + kk
                    b2 = b - 1 + dnk
                    if plot:
                        if expn == 0:
                            plot_distribution(a2, b2, dist_type="beta", ax=ax, shaded=True,  shade_alpha=.8,
                                            plot_kwargs=dict(color="w",  lw=3,
                                                            label="prior"), 
                                            ax_kwargs=dict(xlim=[-0.1, 1.1], 
                                                            ylim=[0, 15],))

                        plot_distribution(a2, b2, dist_type="beta", ax=ax, shaded=True, shade_alpha=.8,
                                                plot_kwargs=dict(color=self.colors[expn+1], lw=2, 
                                                                label=exp), 
                                                ax_kwargs=dict(xlim=[-0.1, 1.1], 
                                                                ylim=[0, 15],
                                                                title=exp))


                    # Plot mean and mode of posterior
                    try:
                        mean =  a2 / (a2 + b2)
                        _mode = (a2 -1)/(a2 + b2 -2)
                        sigmasquared = (a2 * b2)/((a2+b2)**2 * (a2 + b2 + 1)) # https://math.stackexchange.com/questions/497577/mean-and-variance-of-beta-distributions
                        prange = percentile_range(get_parametric_distribution("beta", a2, b2)[1])
                    except: continue
                    modes[exp], means[exp], params[exp], sigmas[exp], pranges[exp] = _mode, mean, ptuple(a2, b2), sigmasquared, prange
                    if plot: ax.axvline(_mode, color=self.colors[expn+1], lw=2, ls="--", alpha=.8)
                    
                elif mode == "hierarhical":
                    pass
                else: raise ValueError(mode)

            if plot:
                ax.set(title="{} bayes".format(mode), ylabel="pdf", xlabel="theta")
                ax.legend()
            return modes, means, params, sigmas, pranges
        else:
            raise ValueError("need to pass either condtions or data")

    def simple_analytical_bayes(self, trials):
        # trials = 1d array of 0s and 1s
        k, n = np.sum(trials), len(trials)
        a2, b2 = self.a - 1 + k, self.b - 1 + n - k
        fact = binom(n, k)

        mean, var, skew, kurt = stats.beta.stats(a2, b2, moments='mvsk')
        return (a2, b2, fact), mean, var
        
    """
        LOGISTIC REGRESSION
    """

    def bayesian_logistic_regression(self, xdata, ydata):
        # ? from doing bayesian data analysis p 324
        print("fitting bayesian logistic regression")
        with pm.Model() as model: 
            # Define priors
            beta0 = pm.Normal('beta0', 0, sd=20)
            beta1 = pm.Normal("beta1", 0, sd=20)

            # Define likelihood
            mu = pm.math.sigmoid(beta0 + beta1*xdata)  # argument is the exponent of the sigmiod function
            likelihood = pm.Bernoulli('y',  mu,  observed=ydata)
            
            # Inference!
            print("inference time")
            trace = pm.sample(1000, tune=500, cores=3, discard_tuned_samples=True)
            pm.traceplot(trace)
        return trace

    def robust_bayesian_logistic_regression(self, xdata, ydata):
        # ? from doing bayesian data analysis p 324
        print("fitting ROBUST bayesian logistic regression")
        with pm.Model() as model: 
            # Define priors
            beta0 = pm.Normal('beta0', 0, sd=30)
            beta1 = pm.Normal("beta1", 0, sd=30)
            
            # Define likelihood
            alpha = pm.Beta("alpha", alpha=1, beta=9)
            mu = alpha * .5 + (1 - alpha) * pm.math.sigmoid(beta0 + beta1*xdata)
            likelihood = pm.Bernoulli('y',  mu,  observed=ydata)


            # Inference!
            print("inference time")
            trace = pm.sample(2000, tune=1000, cores=2, nuts_kwargs={'target_accept': 0.99}, discard_tuned_samples=True)
            
        return trace    