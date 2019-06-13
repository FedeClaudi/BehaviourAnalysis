# %%
from Utilities.imports import *
from Modelling.glm.glm_data_loader import GLMdata
import pymc3 as pm

from scipy.special import logit
from sklearn.model_selection import train_test_split

import statsmodels.formula.api as smf
from patsy import dmatrices
import seaborn as sns
from theano import tensor as tt
%pylab inline
%config InlineBackend.figure_format = 'retina'
%matplotlib inline  

# %% 
# get data
glm = GLMdata(load_trials_from_file=True)

d = namedtuple("d", "all asym sym")
data = d(glm.trials, glm.asym_trials, glm.sym_trials)

# %%
# Extract more info on the data
right_rois, left_rois = [18, 6, 13], [17, 8, 11, 2]


for k,v in data._asdict().items():
    v["mean_expl_speed"] = [np.nanmean(dd.tracking_data_exploration[:, 2]) for i, dd in v.iterrows()]
    v["delta_theta"] = 135 - v.iTheta.values

    # TODO make it cleaner
    v["time_on_left_exp"] = [np.sum(np.isin(dd.tracking_data_exploration[:, -1], left_rois)) for i, dd in v.iterrows()]
    v["time_on_right_exp"] = [np.sum(np.isin(dd.tracking_data_exploration[:, -1], right_rois)) for i, dd in v.iterrows()]

    # TODO this doesnt work
    v= v.drop(v.loc[v["time_on_right_exp"] == 0].index, axis=0)
    v["rel_time_on_right"] = v["time_on_left_exp"].values / v["time_on_right_exp"].values
    v = v.drop([x for x,i in enumerate(np.isnan(v["rel_time_on_right"])) if i])
    v = v.drop([x for x,i in enumerate(np.isinf(v["rel_time_on_right"])) if i])
    v["correct_rel_time_on_right"] = v.rel_time_on_right.values / v.rLen.values


# %%
# split 
train, test = train_test_split(data.sym, test_size=.3)


#%%
# Fit and plot
eq = "escape_right ~ time_on_right_exp + mean_expl_speed + x_pos + mean_expl_speed" 
model, res, y, predictions = glm.run_glm(train, eq)
print(res.summary())

# predict test
y_test = test.escape_right.values.ravel()
predictions_test = res.predict(test).values

plotter2(y, predictions, "train", train.experiment_asymmetric.values)

plotter2(y_test, predictions_test, "test", test.experiment_asymmetric.values)

#%%
# summary
print(res.summary())

# %%
def plotter2(y, predictions, title, exp_label, n_iters=400):
    x = np.arange(len(predictions))
    sort_idxs = np.argsort(predictions)

    y = y[sort_idxs]
    yy = np.zeros_like(y)-.05
    yy[y > 0] = 1.05
    p = predictions[sort_idxs]

    n_obs = len(p)
    f, ax = plt.subplots(figsize=(5.75, 4),facecolor=[.12, .12, .14])

    sim_pred, sim_pred_arr = [], np.zeros((n_iters, n_obs))

    for i, pp in enumerate(p):
        xx = np.ones((n_iters))*i
        sp = np.random.binomial(1, pp, size=n_iters)
        sim_pred_arr[:, i] = sp

    ax.plot(x, np.sort(y), color="purple", alpha=.5, label="obs")
    ax.scatter(x, yy, c=exp_label[sort_idxs], cmap="Reds", alpha=.5, label="obs", vmin=-1)
    ax.scatter(x, p, c=exp_label[sort_idxs], cmap="Greens", alpha=.8, label="pred", vmin=-1)
    ax.legend()
    ax.set(title=title)

    ax.set(ylabel="escape_arm", xlabel="trials", yticks=[0,1], yticklabels=["left", "right"])



#%%
