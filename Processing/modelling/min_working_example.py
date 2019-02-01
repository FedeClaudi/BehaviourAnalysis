import numpy as np
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt



def print_divergence_perc(trace):
    # display the total number and percentage of divergent
    divergent = trace['diverging']
    print('Number of Divergent %d' % divergent.nonzero()[0].size)
    divperc = divergent.nonzero()[0].size / len(trace) * 100
    print('Percentage of Divergent %.1f' % divperc)


def divergence_scatterplot(trace):    
    for i in np.arange(10):
        pm.pairplot(trace,
                sub_varnames=['thetas_{}'.format(i),'phi'],
                divergences=True,
                color='C3', figsize=(10, 5), kwargs_divergence={'color':'C2'})
        plt.title('scatter plot between phi and theta[{}]'.format(i))


def run():
    # Define data
    datasets_names = ['A', 'B']
    number_of_individuals =[22, 17]

    n_trials_A = [21, 15,  6,  5, 10,  6,  4,  6,  5,  7, 14, 12, 15,  4,  4,  6,  6,  9,  7,  6, 11, 10]
    hits_A = [21, 14,  6,  0,  6,  6,  3,  6,  5,  6, 14,  9, 15,  4,  4,  5,  6,  8,  7,  4,  8, 10]

    n_trials_B = [5,  5, 33,  4, 13, 18, 24,  8,  8,  9,  9,  7, 14,  8, 15,  9, 11]
    hits_B = [2,  5, 26,  3,  7,  7, 13,  6,  1,  5,  4,  2,  7,  5,  9,  4,  1]

    datasets = [(number_of_individuals[0], n_trials_A, hits_A), (number_of_individuals[1], n_trials_B, hits_B)]

    # Model each dataset
    for i, (m, n, h) in enumerate(datasets):
        print('Modelling dataset: ', datasets_names[i])

        # pyMC3 model
        with pm.Model() as model:
            # The model is from: https://docs.pymc.io/notebooks/hierarchical_partial_pooling.html

            # Define hyperpriors
            phi = pm.Uniform('phi', lower=0.0, upper=1.0)

            kappa_log = pm.Exponential('kappa_log', lam=1.5)
            kappa = pm.Deterministic('kappa', tt.exp(kappa_log))

            # define second level of hierarchical model
            thetas = pm.Beta('thetas', alpha=phi*kappa, beta=(1.0-phi)*kappa, shape=m)

            # Likelihood
            y = pm.Binomial('y', n=n, p=thetas, observed=h)

            # Fit
            trace = pm.sample(8000, tune=2000, nuts_kwargs={'target_accept': 0.95}) 

        # Show traceplot
        pm.traceplot(trace)
        print_divergence_perc()
        divergence_scatterplot()
        pm.pairplot(trace)
        plt.show()

        A = 1
    plt.show()

        















if __name__ == "__main__":
    run()


