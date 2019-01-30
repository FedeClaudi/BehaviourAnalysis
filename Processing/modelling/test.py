import numpy as np
import scipy.stats as stats
import pymc3 as pm
import matplotlib.pyplot as plt


def run():
    def distribution_visualiser(dist):
        try:
            samples = dist.random(size=200000)
            plt.figure()
            plt.hist(samples, bins=100)
            plt.plot(np.sort(samples))
        except: pass


    # Generate fake data
    """ 
        The data is an m-by-n array containing the outcomes of n trials for m individuals
        The array is generated from a bernoulli distribution, each m individual has a unique p value for the probability distribution
        p values are chosen randomly between 0 and 1
    """
    m = 20
    n = 100
    p_min, p_max = 0.2, .5

    p = np.random.uniform(p_min, p_max, size=m)
    trials = stats.bernoulli.rvs(p=p, size=(n, m))
    # trials = trials.T  # shape = m, n
    index = np.arange(m)


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
        mu_a = pm.Normal('mu_a', mu=0., sd=1e5)
        sigma_a = pm.HalfCauchy('sigma_a', 5)

        # Random intercepts
        a = pm.Normal('a', mu=mu_a, sd=sigma_a, shape=m)

        # Model error
        sigma_y = pm.HalfCauchy('sigma_y', 5)

        # Expected value
        y_hat = a[index]

        # Data likelihood
        y_like = pm.Normal('y_like', mu=y_hat, sd=sigma_y, observed=trials)

        partial_pooling_trace = pm.sample(1000, tune=1000)

    pm.traceplot(partial_pooling_trace)
    plt.show()





if __name__ == "__main__":
    run()
