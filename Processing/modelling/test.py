import numpy as np
import scipy.stats as stats
import pymc3 as pm
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import theano.tensor as tt

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
    n = 10000
    p_min, p_max = 0.2, .5

    p = np.random.uniform(p_min, p_max, size=m)
    trials = stats.bernoulli.rvs(p=p, size=(n, m))
    # trials = trials.T  # shape = m, n
    index = np.arange(m)
    hits = np.sum(trials,0)



    with pm.Model() as model:
        phi = pm.Uniform('phi', lower=0.0, upper=1.0)

        kappa_log = pm.Exponential('kappa_log', lam=1.5)
        kappa = pm.Deterministic('kappa', tt.exp(kappa_log))

        # BoundedNormal = pm.Bound(pm.Normal, lower=-1.0, upper=1.0)
        offset = pm.Normal('offset', mu=0, sd=0.1, shape = m)

        # thetas = pm.Normal('thetas', mu=phi, sd=100, shape=m)
        thetas = pm.Beta('thetas', alpha=phi*kappa, beta=(1.0-phi)*kappa, shape=m)

        offset_thetas = pm.Deterministic("offset_thetas", thetas + offset*kappa)
        y = pm.Binomial('y', n=n, p=thetas, observed=hits)
        ye = pm.Binomial('ye', n=n, p=thetas, shape=m)

        # step = pm.Metropolis()
        partial_pooling_trace = pm.sample(2000, tune=1000, nuts_kwargs={'target_accept': 0.95}) 



    pm.traceplot(partial_pooling_trace)
    plt.show()

def run2():
    # Data of the Eight Schools Model
    J = 8
    y = np.array([28.,  8., -3.,  7., -1.,  1., 18., 12.])
    sigma = np.array([15., 10., 16., 11.,  9., 11., 10., 18.])
    # tau = 25.

    """
        with pm.Model() as Centered_eight:
            mu = pm.Normal('mu', mu=0, sd=5)
            tau = pm.HalfCauchy('tau', beta=5)
            theta = pm.Normal('theta', mu=mu, sd=tau, shape=J)
            obs = pm.Normal('obs', mu=theta, sd=sigma, observed=y)

    """



    with pm.Model() as NonCentered_eight:
        SEED = [20100420, 20100234]

        mu = pm.Normal('mu', mu=0, sd=5)
        tau = pm.HalfCauchy('tau', beta=5)
        theta_tilde = pm.Normal('theta_t', mu=0, sd=1, shape=J)
        theta = pm.Deterministic('theta', mu + tau * theta_tilde)
        obs = pm.Normal('obs', mu=theta, sd=sigma, observed=y)

        with NonCentered_eight:
            fit_ncp90 = pm.sample(5000, chains=2, tune=1000, random_seed=SEED,
                                nuts_kwargs=dict(target_accept=.90))

    pm.traceplot(fit_ncp90)

    # display the total number and percentage of divergent
    divergent = fit_ncp90['diverging']
    print('Number of Divergent %d' % divergent.nonzero()[0].size)

print('Runing on PyMC3 v{}'.format(pm.__version__))




if __name__ == "__main__":
    run()

    plt.show()
