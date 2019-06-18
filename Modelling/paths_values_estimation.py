# %% 
# imports
from Utilities.imports import *
from theano import tensor as tt
import pymc3 as pm
%matplotlib inline
from Modelling.glm.glm_data_loader import GLMdata


# %%
# Analyse distribution of velocity
glm = GLMdata(load_trials_from_file=True)
data = glm.asym_trials
mean_escape_speed = [np.nanmean(tr[:, 2]) for tr in data.tracking_data_exploration.values]
std_escape_speed = [np.nanstd(tr[:, 2]) for tr in data.tracking_data_exploration.values]

n_mice = len(mean_escape_speed)

# print("mean expl speed: ", mean_escape_speed, " +-:", std_escape_speed)

# %%
# model
lenl, lenr = 1.0, 0.728
kappa = 2
lalpha, lbeta = beta_distribution_params(omega=lenl, kappa=kappa)
ralpha, rbeta = beta_distribution_params(omega=lenr, kappa=kappa)

mean_speed = 2

with pm.Model() as glmm1:
    # Noisy estimate of path lenghts
    x_l = pm.Beta("x_l", alpha=lalpha, beta=kappa, shape=n_mice)
    x_r = pm.Beta("x_r", alpha=ralpha, beta=rbeta, shape=n_mice)

    # Speed
    speed = pm.Normal("speed", np.nanmean(mean_escape_speed), np.nanmean(std_escape_speed), 
                        shape=n_mice, observed=mean_escape_speed)

    # Estimate the time necessary
    t_l_est = pm.Deterministic("t_l_est", x_l/speed)
    t_r_est = pm.Deterministic("t_r_est", x_r/speed)

    # Estimate difference between the two arms
    rt = pm.Deterministic("rt", t_l_est/t_r_est)
    
    trace = pm.sample(1000, tune=1000)
    
# pm.traceplot(trace)
dftrace = pm.trace_to_dataframe(trace)

# %%
rt_cols = [c for c in dftrace.columns if "rt" in c]
outcomes = []

for c in rt_cols:
    outcomes.append(np.mean([1 if x<1 else 0 for x in dftrace[c].values]))

np.mean(outcomes)    
# print("length right: {} - length left: {} - taking right: {}".format(lenl, lenr, np.mean(outcomes)))

#%%
