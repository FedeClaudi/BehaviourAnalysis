# %%
from Utilities.imports import *
from Modelling.glm.glm_data_loader import GLMdata
%matplotlib inline  


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

# %% 
# Get mean speed during exploration
for d in [glm.trials, glm.asym_trials, glm.sym_trials]:
    d["mean_expl_speed"] = [np.nanmean(dd.tracking_data_exploration[:, 2]) for i, dd in d.iterrows()]

#%%
# fit the glm
eq = "escape_right ~ rLen*iTheta*mean_expl_speed"
# eq = "escape_right ~ rLen -1"
# eq = "escape_right ~ x_pos + y_pos + tot_time_in_shelter + iTheta + rLen + median_vel +  max_speed + origin_right + session_number_trials + speed + time_out_of_t"
# 

glm.results = {}
for name, dd in zip(["all", "asym", "sym"], [glm.trials, glm.asym_trials, glm.sym_trials]):
    model, res, y, predictions = glm.run_glm(dd, eq)
    glm.results[name] = (res, y, predictions, dd.experiment_asymmetric.values)

    # ? plot the results
    # color_label = dd.experiment_asymmetric.values+1
    # plotter(y, predictions, color_label, logistic=False)

    print(res.summary())

# ! plot results
for exp, (res, y, pred, exp_label) in glm.results.items():
    correct = []
    for i in range(10000):
        pp = np.random.binomial(1, pred)
        correct.append(np.sum((pp == y)))
    print("Correct estimates: {} - mean: {} out of {} +- {}".format(exp, np.round(np.mean(correct), 2), len(y), round(np.std(correct), 2)))

    plotter2(y, pred, exp, exp_label)
    break 

#%%
# print results summary

for exp, (res, y, pred, _) in glm.results.items():
    # print("\nExp: ", exp)
    # print(res.summary())
    if exp == "all":
        summary = pd.DataFrame(res.params, columns=["all"])
    else:
        summary[exp] = res.params
summary
#%%
# ! plotting functions



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
    f, axarr = plt.subplots(figsize=(12, 8), ncols=2, facecolor=[.12, .12, .14])

    sim_pred, sim_pred_arr = [], np.zeros((n_iters, n_obs))

    for i, pp in enumerate(p):
        xx = np.ones((n_iters))*i
        sp = np.random.binomial(1, pp, size=n_iters)
        # axarr[0].scatter(xx, sp, color='k', alpha=0.01 )
        sim_pred.append(np.mean(sp))
        sim_pred_arr[:, i] = sp

    axarr[0].scatter(x, np.sort(y), c=exp_label[sort_idxs], cmap="Purples", alpha=.5, label="obs", vmin=-1)
    axarr[0].scatter(x, yy, c=exp_label[sort_idxs], cmap="Reds", alpha=.5, label="obs", vmin=-1)
    axarr[0].scatter(x, p, c=exp_label[sort_idxs], cmap="Greens", alpha=.8, label="pred", vmin=-1)
    axarr[0].legend()
    axarr[0].set(title="title")

    # TODO         make padded convolve work

    ww = 31
    for i in range(1000):
        axarr[1].plot(np.convolve(np.random.binomial(1, p).ravel(), np.ones(ww,dtype=int),'valid')/ww, color="k", alpha=.1,  label="pred")

    axarr[1].plot(np.convolve(y, np.ones(ww,dtype=int),'valid')/ww, lw=5, alpha=1, color="r")
    axarr[1].plot(np.convolve(np.sort(y[np.argsort(exp_label)]), np.ones(ww,dtype=int),'valid')/ww, lw=5, color="m", alpha=0.8)

    axarr[1].set(title="Windows sum- window: {}".format(ww))

    for ax in axarr:
        ax.set(ylabel="escape_arm", xlabel="trials", yticks=[0,1], yticklabels=["left", "right"])


# plotter2(y, predictions, "")




#%%
