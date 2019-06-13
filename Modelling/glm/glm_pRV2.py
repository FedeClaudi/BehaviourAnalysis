# %%
from Utilities.imports import *
from Modelling.glm.glm_data_loader import GLMdata
import pymc3 as pm

from scipy.special import logit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler as Scaler

import statsmodels.formula.api as smf
from patsy import dmatrices
import seaborn as sns
from theano import tensor as tt
%pylab inline
%config InlineBackend.figure_format = 'retina'
%matplotlib inline  

# %% 
# get data
glm = GLMdata(load_trials_from_file=False)

d = namedtuple("d", "all asym sym")
data = d(glm.trials, glm.asym_trials, glm.sym_trials)

# %%
# Extract more info on the data
right_rois, left_rois = [18, 6, 13], [17, 8, 11, 2, 3, 12, 22]

# Remove trials from sessinos with insufficient exploration
cleaned = []
for k,v in data._asdict().items():
    cleaned.append(v.loc[(v.total_travel > 10000) & (v.tot_time_in_shelter < 650)])
data = d(*cleaned)


# %%
# Extract metrics from expl
for k,v in data._asdict().items():
    v["mean_expl_speed"] = [np.nanmean(dd.tracking_data_exploration[:, 2]) for i, dd in v.iterrows()]
    v["delta_theta"] = 135 - v.iTheta.values

    # TODO make it cleaner
    v["time_on_left_exp"] = [np.sum(np.isin(dd.tracking_data_exploration[:, -1], left_rois)) for i, dd in v.iterrows()]
    v["time_on_right_exp"] = [np.sum(np.isin(dd.tracking_data_exploration[:, -1], right_rois)) for i, dd in v.iterrows()]

    # TODO this doesnt work
    v["rel_time_on_right"] = v["time_on_left_exp"].values / v["time_on_right_exp"].values
    v["correct_rel_time_on_right"] = v.rel_time_on_right.values / v.rLen.values

# %% 
# Etxtract metrics from out tracking
for k,v in data._asdict().items():
    v["out_trip_duration"] = [dd.outward_tracking_data.shape[0] for i, dd in v.iterrows()]
    v["out_trip_mean_speed"] = [np.mean(dd.outward_tracking_data[:, 2]) for i,dd in v.iterrows()]
    v["out_trip_iAngVel"]  = [np.sum(np.abs(calc_ang_velocity(calc_angle_between_points_of_vector(dd.outward_tracking_data[:, :2]))))/dd.out_trip_duration
                                 for i,dd in v.iterrows()]

# %%
# drop columns
cols_to_keep = ['session_uid',  'time_out_of_t', 'x_pos', 'experiment_asymmetric',
       'y_pos', 'speed',  'escape_right', 
       'origin_right',
       'total_travel', 'tot_time_in_shelter', 'tot_time_on_threat', 
       'median_vel', 'rLen', "out_trip_iAngVel", 
       'iTheta', 'mean_expl_speed', 'delta_theta', 'time_on_left_exp', "out_trip_mean_speed", 
       'time_on_right_exp', 'rel_time_on_right', 'correct_rel_time_on_right', "out_trip_duration"]

kept = []
for k,v in data._asdict().items():
    kept.append(v[cols_to_keep])
data = d(*kept)

# %%
# Normalize columns
cols_to_norm = ['time_out_of_t', 'x_pos',
       'y_pos', 'speed',  
       'total_travel', 'tot_time_in_shelter', 'tot_time_on_threat', 
       'median_vel', 'rLen',
       'iTheta', 'mean_expl_speed', 'delta_theta', 'time_on_left_exp',
       'time_on_right_exp', 'rel_time_on_right', 'correct_rel_time_on_right']
scaler = Scaler()
normalized = []
for k,v in data._asdict().items():
    for col in cols_to_norm:
        v[col] = scaler.fit_transform(v[col].values.reshape(-1, 1))

# %%
# Inspect data
pd.plotting.scatter_matrix(data.all)

# %%
# split 
train, test = train_test_split(data.all, test_size=.4)


#%%
# Fit and plot
eq = "escape_right ~ rLen + correct_rel_time_on_right + iTheta  + mean_expl_speed + session_uid + speed + time_out_of_t + tot_time_on_threat + tot_time_in_shelter + total_travel + x_pos" 

model, res, y, predictions = glm.run_glm(data.all, eq, regularized=False)
print(res.summary())

# predict test
y_test = test.escape_right.values.ravel()
predictions_test = res.predict(test).values

# plot
plotter3(y, predictions, "train",data.all.experiment_asymmetric.values)
# plotter3(y_test, predictions_test, "test", test.experiment_asymmetric.values)

# test
for yy, pp, t in zip((y, y_test), (predictions, predictions_test), ("train", "test")):
    correct = []
    for i in range(10000):
        p = np.random.binomial(1, pp)
        correct.append(np.sum((p ==yy)))
    c, n = np.round(np.mean(correct), 2),  len(yy)
    print("Correct estimates: {} - mean: {} of {} +- {} - {}% correct".format(t, c,n, round(np.std(correct), 2), np.round(c/n, 2)*100))

# %%
mi, ma  = np.argmin(predictions), np.argmax(predictions)
mima =pd.DataFrame([train.iloc[mi], train.iloc[ma]]).T
mima

#%%
# summary
print(res.summary())
print(res.params)
# %%
def plotter2(y, predictions, title, exp_label, n_iters=400):
    x = np.arange(len(predictions))
    sort_idxs = np.argsort(predictions)

    y = y[sort_idxs]
    yy = np.zeros_like(y)-.05
    yy[y > 0] = 1.05
    p = predictions[sort_idxs]

    n_obs = len(p)
    f, axarr = plt.subplots(figsize=(5.75, 4),facecolor=[.12, .12, .14], ncols=2)
    ax = axarr[0]
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

    ax = axarr[1]
    ax.plot(np.cumsum(np.sort(y)), color="purple")
    ax.plot(np.cumsum(yy), color="red")
    ax.plot(np.cumsum(p), color="green")




#%%
def plotter3(y, predictions, title, exp_label, n_iters=400):
    x = np.arange(len(predictions))
    sort_idxs = np.argsort(y)

   
    p = predictions[sort_idxs]
    y = y[sort_idxs]

    f, axarr = plt.subplots(figsize=(5.75, 4),facecolor=[.12, .12, .14], ncols=2)

    ax = axarr[0]
    ax.plot(x, y, color="purple", alpha=.5, label="obs")
    ax.scatter(x, p, c=exp_label[sort_idxs], cmap="Greens", alpha=.8, label="pred", vmin=-1)
    ax.legend()
    ax.set(title=title)
    ax.set(ylabel="escape_arm", xlabel="trials", yticks=[0,1], yticklabels=["left", "right"])

    ax = axarr[1]
    ax.plot(np.cumsum(np.sort(y)), color="purple")
    ax.plot(np.cumsum(p), color="green")


#%%
