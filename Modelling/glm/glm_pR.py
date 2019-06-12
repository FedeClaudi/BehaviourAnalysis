# %%
from Utilities.imports import *
from Modelling.glm.glm_data_loader import GLMdata
#%%
"""
['duration',
 'escape_arm',
 'escape_duration',
 'escape_left',
 'escape_right',
 'experiment_asymmetric',
 'experiment_name',
 'experiment_symmetric',
 'exploration_id',
 'exploration_start',
 'fps',
 'iTheta',
 'max_speed',
 'mean_speed',
 'median_vel',
 'origin_arm',
 'origin_arm',
 'origin_left',
 'origin_right',
 'rLen',
 'session_number_trials',
 'session_uid',
 'speed',
 'time_out_of_t',
 'tot_time_in_shelter',
 'tot_time_on_threat',
 'tot_time_in_shelter',
 'total_travel',
 'tracking_data_exploration',
 'tracking_data_trial',
 'trial_id',
 'x_pos',
 'y_pos']
"""
# %% 
# get data
glm = GLMdata(load_trials_from_file=True)
data = glm.trials
features = sorted(data.columns)
data_descr = data.dtypes
#%%
# fit the glm
eq = "escape_right ~ x_pos + y_pos + tot_time_in_shelter + iTheta + rLen + median_vel +  max_speed + origin_right + session_number_trials + speed + time_out_of_t"
glm.results = {}
for name, dd in zip(["all", "asym", "sym"], [glm.trials, glm.asym_trials, glm.sym_trials]):
    model, res, y, predictions = glm.run_glm(dd, eq)
    glm.results[name] = res
    # plot the results
    color_label = dd.experiment_asymmetric.values+1
    plotter(y, predictions, color_label, logistic=False)

    print(res.summary())

# %% quantify results
# TODO make this work for all experiments

# iterate N time, get the number of correct predictions at ever iter given our predictions
correct = []
for i in range(10000):
    pp = np.random.binomial(1, predictions)
    correct.append(np.sum((pp == y)))

print("Correct estimates: - mean: {} out of {} +- {}".format(np.round(np.mean(correct), 2), len(y), round(np.std(correct), 2)))



#%%
def plotter(y, predictions, label, logistic=False):
        x = np.arange(len(predictions))
        sort_idxs_p = np.argsort(predictions)
        sort_idxs_y = np.argsort(y)

        yy = np.zeros_like(y)-.1
        yy[y > 0] = 1.1

        f, axarr = plt.subplots(figsize=(9, 8), ncols=2)

        for ax, sort_idxs, title in zip(axarr, [sort_idxs_y, sort_idxs_p], ["sort Y", "sort Pred"]):
            ax.scatter(x, y[sort_idxs], c=label[sort_idxs], cmap="Reds", label = 'Obs', alpha=.5, vmin=0)

            ax.scatter(x, predictions[sort_idxs],  c=label[sort_idxs], cmap="Greens", label = 'Pred', alpha=.75, vmin=0)

            if logistic:
                sns.regplot(x, predictions[sort_idxs], logistic=True, 
                                            truncate=True, scatter=False, ax=ax)

            ax.set(title = title, ylabel="escape_arm", xlabel="trials", yticks=[0,1], yticklabels=["left", "right"])
            ax.legend()


#%%
def plotter2(y, predictions, n_iters=100):
    x = np.arange(len(predictions))
    sort_idxs = np.argsort(predictions)

    y = y[sort_idxs]
    yy = np.zeros_like(y)-.05
    yy[y > 0] = 1.05
    p = predictions[sort_idxs]

    n_obs = len(p)
    f, axarr = plt.subplots(figsize=(12, 8), ncols=3, facecolor=[.12, .12, .14])
    
    sim_pred, sim_pred_arr = [], np.zeros((n_iters, n_obs))

    for i, pp in enumerate(p):
        xx = np.ones((n_iters))*i
        sp = np.random.binomial(1, pp, size=n_iters)
        axarr[0].scatter(xx, sp, color='k', alpha=0.01 )
        sim_pred.append(np.mean(sp))
        sim_pred_arr[:, i] = sp

    axarr[0].scatter(x, yy, color="r", alpha=.5, label="obs")
    axarr[0].scatter(x, p, color="g", alpha=.8, label="pred")
    axarr[0].legend()

    sns.regplot(x, y, logistic=True, color="r", label="obs", ax=axarr[1])
    sns.regplot(x, sim_pred_arr, logistic=True, color="k", label="pred", ax=axarr[1])
    axarr[1].legend()

    ww = 21
    for i in range(100):
        axarr[2].plot(np.convolve(np.random.binomial(1, p).ravel(), np.ones(ww,dtype=int),'valid'), color="g", alpha=.15,  label="pred")

    axarr[2].plot(np.convolve(y,np.ones(ww,dtype=int),'valid'), color="r")
    axarr[2].set(title="Windows sum- window: {}".format(ww), facecolor=[.05, .05, .05])


plotter2(y, predictions)

#%%

#%%
x = np.arange(len(predictions))
sort_idxs = np.argsort(predictions)

y = y[sort_idxs]
yy = np.zeros_like(y)-.05
yy[y > 0] = 1.05
p = predictions[sort_idxs]

n_obs = len(p)
f, axarr = plt.subplots(figsize=(12, 8), ncols=3, facecolor=[.12, .12, .14])

sim_pred, sim_pred_arr = [], np.zeros((n_iters, n_obs))

for i, pp in enumerate(p):
    xx = np.ones((n_iters))*i
    sp = np.random.binomial(1, pp, size=n_iters)
    axarr[0].scatter(xx, sp, color='k', alpha=0.01 )
    sim_pred.append(np.mean(sp))
    sim_pred_arr[:, i] = sp

axarr[0].scatter(x, yy, color="r", alpha=.5, label="obs")
axarr[0].scatter(x, p, color="g", alpha=.8, label="pred")
axarr[0].legend()

sns.regplot(x, y, logistic=True, color="r", label="obs", ax=axarr[1])
xx = np.tile(x, n_iters).reshape(n_iters, n_obs)
sns.regplot(xx, sim_pred_arr, logistic=True, color="k", label="pred", ax=axarr[1])
axarr[1].legend()

ww = 21
for i in range(100):
    axarr[2].plot(np.convolve(np.random.binomial(1, p).ravel(), np.ones(ww,dtype=int),'valid'), color="g", alpha=.15,  label="pred")

axarr[2].plot(np.convolve(y,np.ones(ww,dtype=int),'valid'), color="r")
axarr[2].set(title="Windows sum- window: {}".format(ww), facecolor=[.05, .05, .05])


#%%
