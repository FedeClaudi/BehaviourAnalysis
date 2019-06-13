# %%
from Utilities.imports import *
from Modelling.glm.glm_data_loader import GLMdata
import pymc3 as pm

from scipy.special import logit

import statsmodels.formula.api as smf
from patsy import dmatrices
import seaborn as sns
from theano import tensor as tt
%pylab inline
%config InlineBackend.figure_format = 'retina'
%matplotlib inline  

# %% 
# get data
glmm = GLMdata(load_trials_from_file=True)

# Get mean speed during exploration
# TODO add to GLM class
for d in [glmm.trials, glmm.asym_trials, glmm.sym_trials]:
    d["mean_expl_speed"] = [np.nanmean(dd.tracking_data_exploration[:, 2]) for i, dd in d.iterrows()]



glmm.trials["delta_theta"] = 135 - glmm.trials.iTheta.values # TODO

# %%
# Get time spent on each arm during expl
# TODO add to class

right_rois, left_rois = [18, 6, 13], [17, 8, 11, 2]

glmm.trials["time_on_left_exp"] = [np.sum(np.isin(dd.tracking_data_exploration[:, -1], left_rois)) for i, dd in glmm.trials.iterrows()]
glmm.trials["time_on_right_exp"] = [np.sum(np.isin(dd.tracking_data_exploration[:, -1], right_rois)) for i, dd in glmm.trials.iterrows()]
glmm.trials= glmm.trials.drop(glmm.trials.loc[glmm.trials["time_on_right_exp"] == 0].index, axis=0)
glmm.trials["rel_time_on_right"] = glmm.trials["time_on_left_exp"].values / glmm.trials["time_on_right_exp"].values

glmm.trials = glmm.trials.drop([x for x,i in enumerate(np.isnan(glmm.trials["rel_time_on_right"])) if i])

# %%
glmm.trials = glmm.trials.drop([x for x,i in enumerate(np.isnan(glmm.trials["rel_time_on_right"])) if i])
glmm.trials = glmm.trials.drop([x for x,i in enumerate(np.isinf(glmm.trials["rel_time_on_right"])) if i])

glmm.trials["correct_rel_time_on_right"] = glmm.trials.rel_time_on_right.values / glmm.trials.rLen.values

data = glmm.trials
cols = data.columns

# %% 
# split
from sklearn.model_selection import train_test_split

# train, test  = glmm.split_dataset("trials", fraction=.3) # TODO fix
train, test = train_test_split(glmm.trials, test_size=.3)


#%%
# ! try fitting a GLM first
"""
155/197
eq = "escape_right ~   rLen + correct_rel_time_on_right + duration + iTheta + max_speed + mean_expl_speed + mean_speed + session_uid + speed + time_out_of_t + tot_time_on_threat + tot_time_in_shelter + total_angular_displacement + total_travel + x_pos + y_pos" 
153/197
eq = "escape_right ~   rLen + correct_rel_time_on_right + iTheta + mean_expl_speed + session_uid + speed + tot_time_on_threat + tot_time_in_shelter + total_travel + x_pos + y_pos" 
131/197
eq = "escape_right ~   rLen + correct_rel_time_on_right + iTheta + mean_expl_speed  + speed + tot_time_on_threat + tot_time_in_shelter + total_travel + x_pos + y_pos" 

"""
eq = "escape_right ~   rLen + correct_rel_time_on_right + duration + iTheta + mean_expl_speed + session_uid + speed + time_out_of_t" 
model, res, y, predictions = glmm.run_glm(train, eq)
print(res.summary())

# predict test
y_test = test.escape_right.values.ravel()
# predictions_test = res.predict(test).values

plotter2(y, predictions, "train", train.experiment_asymmetric.values)

# plotter2(y_test, predictions_test, "test", test.experiment_asymmetric.values)
correct = []
for i in range(10000):
        p = np.random.binomial(1, predictions)
        correct.append(np.sum((p ==y)))
c, n = np.round(np.mean(correct), 2),  len(y)
print("Correct estimates: {} - mean: {} of {} +- {} - {}% correct".format("f", c,n, round(np.std(correct), 2), np.round(c/n, 2)*100))


# %%
for yy, pp, t in zip((y, y_test), (predictions, predictions_test), ("train", "test")):
    correct = []
    for i in range(10000):
        p = np.random.binomial(1, pp)
        correct.append(np.sum((p ==yy)))
    c, n = np.round(np.mean(correct), 2),  len(yy)
    print("Correct estimates: {} - mean: {} of {} +- {} - {}% correct".format(t, c,n, round(np.std(correct), 2), np.round(c/n, 2)*100))


# f,ax = plt.subplots()
# ax.hist(np.array(correct)/len(predictions))
# ax.axvline(np.sum(y > .5)/len(predictions), color="r")
# ax.axvline(.5, color="k")

# %% 
# Prepare matrixes for fixed predictors
# formula = "escape_right ~ rLen + iTheta + correct_rel_time_on_right + rLen" 
Y, X = dmatrices(eq, data=glmm.trials, return_type='matrix')
terms  = X.design_info.column_names

# prep matrix for mixed effect
_, Z   = dmatrices('escape_right ~ session_uid', data=glmm.trials, return_type='matrix')

X      = np.asarray(X) # fixed effect
Y      = np.asarray(Y).flatten()
Z      = np.asarray(Z) # mixed effect

nfixed = np.shape(X)
nrandm = np.shape(Z)


# %%
# Fit PyMC3 model
# beta0     = np.linalg.lstsq(X,Y)  # Solves the equation a x = b by computing a vector x that minimizes the Euclidean 2-norm || b - a x ||^2. 
# ! Estimate parameters with PyMC3
with pm.Model() as glmm1:
    # Fixed effect
    beta = pm.Normal('beta', mu=0., sd=100., shape=(nfixed[1]))

    # mixed effect
    s    = pm.HalfCauchy('s', 10.)
    b    = pm.Normal('b', mu=0., sd=s, shape=(nrandm[1]))
    eps  = pm.HalfCauchy('eps', 5.)

    # linear estmator
    mu_est = pm.Deterministic('mu_est', tt.dot(X,beta)+tt.dot(Z,b))

    # fit the observed data
    p = pm.Normal('p', mu_est, eps, observed=Y)

    # est data
    p_est = pm.Normal("p_est", mu_est, eps, shape=nfixed[0])

    trace = pm.sample(500, tune=800)
    
pm.traceplot(trace)
df_trace = pm.trace_to_dataframe(trace)


# %%
# ? check model

fixed_pymc = pm.summary(trace, varnames=['beta'])
b = np.asarray(fixed_pymc['mean'])
randm_pymc = pm.summary(trace, varnames=['b'])
q = np.asarray(randm_pymc['mean'])

fitted = np.dot(X,b).flatten()+ np.dot(Z,q).flatten()

# summary
for n,v in zip(terms, b):
    print(n,": ", round(v,2))


f, ax = plt.subplots(figsize=(4,4))

sort_idx = np.argsort(fitted)

ax.plot(Y[sort_idx],'o',color='r', label = 'Obs', alpha=1)
ax.plot(fitted[sort_idx], lw=3, label="Model ", alpha=.8)

ax.legend()


#%%
good_traces_names = [p for p in df_trace if "mu_est" in p]
estimates = df_trace[good_traces_names]
estimates = estimates.median()

#%%
f, axarr = plt.subplots(ncols=2)

sort_idx = np.argsort(estimates.values)
y_sort = np.argsort(Y)

axarr[0].plot(Y[sort_idx],'o',color='r', label = 'Obs', alpha=1)
axarr[1].plot(Y[y_sort], color='m', label = 'Obs', alpha=1)
axarr[0].plot(estimates.values[sort_idx], 'o', color="g")
axarr[1].plot(estimates.values[y_sort],  color="g")



#%%
def plotter2(y, predictions, title, exp_label, n_iters=400):
    def convolve(a):
        padded = np.pad(a, (int(np.ceil(ww/2))+4), 'constant', constant_values=(0))[5:-5]
        convolved = np.convolve(padded, np.ones(ww,dtype=int),'valid')/ww
        return convolved

    x = np.arange(len(predictions))
    sort_idxs = np.argsort(predictions)

    y = y[sort_idxs]
    yy = np.zeros_like(y)-.05
    yy[y > 0] = 1.05
    p = predictions[sort_idxs]

    n_obs = len(p)
    f, axarr = plt.subplots(figsize=(5.75, 4), ncols=2, facecolor=[.12, .12, .14])

    sim_pred, sim_pred_arr = [], np.zeros((n_iters, n_obs))

    for i, pp in enumerate(p):
        xx = np.ones((n_iters))*i
        sp = np.random.binomial(1, pp, size=n_iters)
        # axarr[0].scatter(xx, sp, color='k', alpha=0.01 )
        sim_pred.append(np.mean(sp))
        sim_pred_arr[:, i] = sp

    axarr[0].plot(x, np.sort(y), color="purple", alpha=.5, label="obs")
    axarr[0].scatter(x, yy, c=exp_label[sort_idxs], cmap="Reds", alpha=.5, label="obs", vmin=-1)
    axarr[0].scatter(x, p, c=exp_label[sort_idxs], cmap="Greens", alpha=.8, label="pred", vmin=-1)
    axarr[0].legend()
    axarr[0].set(title="title")

    # TODO         make padded convolve work

    ww = 5
    # for i in range(1000):
    #     axarr[1].plot(np.convolve(np.random.binomial(1, p).ravel(), np.ones(ww,dtype=int),'valid')/ww, color="k", alpha=.1,  label="pred")

    # axarr[1].plot(np.convolve(y, np.ones(ww,dtype=int),'valid')/ww, lw=5, alpha=1, color="r")
    # axarr[1].plot(np.convolve(np.sort(y[np.argsort(exp_label)]), np.ones(ww,dtype=int),'valid')/ww, lw=5, color="m", alpha=0.8)
    # axarr[1].plot(line_smoother(y,order=2), lw=5, alpha=1, color="r")
    # axarr[1].plot(line_smoother(np.sort(y[np.argsort(exp_label)]), order=2),  lw=5, color="m", alpha=0.8)

    # axarr[1].set(title="Windows sum- window: {}".format(ww))

    for ax in axarr:
        ax.set(ylabel="escape_arm", xlabel="trials", yticks=[0,1], yticklabels=["left", "right"])



#%%
