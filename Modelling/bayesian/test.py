# %%
# Imports
import sys
sys.path.append("./")

from Utilities.imports import *
from Processing.analyse_experiments import ExperimentsAnalyser

%matplotlib inline

# %%
ea = ExperimentsAnalyser()
data = ea.load_trials_from_pickle()

# %%
# Define hyper priors
mu = stats.beta(1, 1)
eta = stats.gamma(.01, .01)

# alpha = mu*eta
# beta = mu*(1-eta)

def alpha(mu, eta):
    return mu.rvs()*eta.rvs()
def beta(mu, eta):
    return mu.rvs()*(1-eta.rvs())

# %%
# Get hits and n trials
for exp, trials in data.items():
    N, K = [], []
    sessions = sorted(set(trials.session_uid.values))
    for uid in sessions:
        sess_trials = trials.loc[trials.session_uid == uid]
        N.append(len(sess_trials))
        K.append(len([i for i,t in sess_trials.iterrows() if "right" in t.escape_arm.lower()]))

    # Calculate the posterior conditional probability p(theta | alpha, beta, y)
    f, ax = plt.subplots()
    for i in range(500):
        al, bt = alpha(mu, eta), beta(mu, eta)

        c, a, b = 1, 1, 1
        for n,y in zip(N, K):
            a += al + y - 1 # setting alpha = 2 for now
            b += bt + n - y - 1 # setting beta = 2 for now

        plot_distribution(a, b, dist_type="beta", xlim=[0, 1], ylim=[0, 10], color="r", alpha=.1, title=exp, ax=ax)
        

#%%
