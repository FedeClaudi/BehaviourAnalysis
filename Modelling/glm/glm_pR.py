# %%
from Utilities.imports import *
from theano import tensor as tt
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import dmatrices 
import pymc3 as pm

from sklearn.preprocessing import RobustScaler as Scaler

from Processing.trials_analysis.all_trials_loader import Trials
from statsmodels.graphics.api import abline_plot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

%matplotlib inline
# %%
def plotter(y, predictions, label):
    x = np.arange(len(predictions))
    sort_idxs_p = np.argsort(predictions)
    sort_idxs_y = np.argsort(y)

    f, axarr = plt.subplots(figsize=(9, 8), ncols=2)

    for ax, sort_idxs, title in zip(axarr, [sort_idxs_y, sort_idxs_p], ["sort Y", "sort Pred"]):
        ax.scatter(x, y[sort_idxs], c=label[sort_idxs], cmap="Reds", label = 'Obs', alpha=.5, vmin=0)
        ax.scatter(x, predictions[sort_idxs],  c=label[sort_idxs], cmap="Greens", label = 'Pred', alpha=.75, vmin=0)

        # sns.regplot(x, predictions[sort_idxs], logistic=True, 
        #                             truncate=True, scatter=False, ax=ax)
        ax.set(title = title, ylabel="escape_arm", xlabel="trials", yticks=[0,1], yticklabels=["left", "right"])
        ax.legend()

# %%
# ? Load bayes posteriors
if sys.platform == "darwin":
    f = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/analysis_metadata/bayes/results/hierarchical_v2.pkl"
    parfile = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/analysis_metadata/maze/escape_paths_features.yml"
    trials_file = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/analysis_metadata/saved_dataframes/glm_data.pkl"
else:
    f = None  # TODO
    parfile = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\maze\\escape_paths_features.yml"
    trials_file = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\saved_dataframes\\glm_data.pkl"



data = pd.read_pickle(f)
cols_to_drop = [c for c in data.columns if "asym_prior" not in c and "sym_prior" not in c]
data = data.drop(cols_to_drop, axis=1)


pRs = data.mean()
#%%
# ? Load the maze design params
params = load_yaml(parfile)

# convert them to df
temp = dict(
    name = [],
    length = [],
    rLen = [],
    iLen = [],
    uLen = [],
    theta_start = [],
    theta_tot = []
)

for exp, pps in params.items():
    for name, ps in pps.items():
        good_name = exp+"_"+name
        temp['name'].append(good_name)
        for i, (dict_list, val) in enumerate(zip(list(temp.values())[1:], ps)):
            dict_list.append(val)

params = pd.DataFrame.from_dict(temp)
#%%
# ? merge the datasets
rL, tt = [], []
for name, p in zip(pRs.index, pRs):
    if "asym" in name:
        sarm = params.iloc[1]
        larm = params.iloc[0]

    else:
        sarm = params.iloc[7]
        larm = params.iloc[7]

    rL.append(sarm.rLen)
    tt.append(larm.theta_tot)

data = pd.DataFrame(pRs)
data.columns = ["pR"]
data["rLen"] = rL
data["iTheta"] = tt


################################################################################################################################################
#######################################         # ! GLM on mouse pR
################################################################################################################################################

#%%
# ? inspect the data
pd.plotting.scatter_matrix(trials, figsize=(16,12), s=1000)

# TODO split dataset into training and test
# train, test = train_test_split(trials, test_size=0.5)


#%%
# ! GLM with stats model
# Create and fit the model
model = smf.glm(formula = "pR ~  iTheta + rLen + ", data=data,
                    family=sm.families.Gaussian(link=sm.families.links.identity))
res = model.fit_regularized()
print(res.params)

# ? plot the fitted model's results
y = data.pR.ravel()
predictions = model.predict(res.params)
plotter(y,  predictions)

mse = mean_squared_error(y, predictions)
print("\nMSE: ", round(mse, 2))

# TODO color obs and pred data by experiment 

################################################################################################################################################
#######################################         # ! GLM on single trials
################################################################################################################################################


# %%
# ? load trials data
trials = pd.read_pickle(trials_file)
# trials = trials.drop("origin_left", axis=1)
# trials["origin_arm_clean"] = ["right" if "Right" in arm[0] else "left" for arm in trials['origin_arm'].values]
# trials = pd.get_dummies(trials, columns=["origin_arm_clean"], prefix=["origin"])

print(trials.columns)

# drop columns
# cols_to_keep = ["tracking_data", "escape_right", "origin_right", "experiment_asymmetric", "time_out_of_t"]
# cols_to_drop = [c for c in trials.columns if c not in cols_to_keep]
# trials = trials.drop(cols_to_drop, axis=1)

# Get more tracking data info
trials["x_pos"] = [t[0, 0] for t in trials.tracking_data.values]
trials["y_pos"] = [t[0, 1] for t in trials.tracking_data.values]
trials["speed"] = [t[0, 2] for t in trials.tracking_data.values]

trials = trials.drop("tracking_data", axis=1)

# ? Expand dataset
rLen, iTheta = [], []
for i, trial in trials.iterrows():
    if trial.experiment_asymmetric:
        sarm = params.iloc[1]
        larm = params.iloc[0]

    else:
        sarm = params.iloc[7]
        larm = params.iloc[7]

    rLen.append(sarm.rLen)
    iTheta.append(larm.theta_tot)

trials['rLen'] = rLen
trials['iTheta'] = iTheta

# %%
# Keep only trials form one experiment
sym_trials = trials.loc[trials.experiment_asymmetric == 0]
asym_trials = trials.loc[trials.experiment_asymmetric == 1]


#%%
# ! GLM to predict arm of escape
# TODO add metrics of exploration: mean speed, time on each arm
# TODO add sessions metrics: number of trials per mouse
# TODO add mouse id? -> maybe need a GLMM for this
# TODO add time to trave to shelter based on different paths and velocity distribution of the mosue

# Create and fit the model
for exp, tts in zip(["all", "asym", "sym"], [trials, asym_trials, sym_trials]):
    print("\n\n{}\n\n".format(exp))
    model = smf.glm(formula = "escape_right ~  time_out_of_t + speed + iTheta + rLen + mean_speed", 
                        data=tts,
                        family=sm.families.Binomial(link=sm.families.links.logit))
    res = model.fit()
    print(res.params)

    # ? plot the fitted model's results
    y = tts.escape_right.ravel()
    predictions = model.predict(res.params)
    plotter(y,  predictions, tts.experiment_asymmetric.values.astype(np.int16)+1)

    mse = mean_squared_error(y, predictions)
    print("\nMSE: ", round(mse, 2))
    # break

#%%
# ? Focus on asym
train, test = train_test_split(asym_trials, test_size=0.33)
eq = "escape_right ~  time_out_of_t + speed + iTheta + rLen + mean_speed + escape_duration + total_angular_displacement"



model = smf.glm(formula = eq, 
                    data=train,
                    family=sm.families.Binomial(link=sm.families.links.logit))
res = model.fit()
print(res.params)

# ? plot the fitted model's results
y = test.escape_right.ravel()

exog = sm.add_constant(test[["time_out_of_t", "speed", "iTheta", "rLen", "mean_speed", "escape_duration", "total_angular_displacement"]])
# TODO prediction brocken
predictions = model.predict(res.params, exog)
plotter(y,  predictions, test.experiment_asymmetric.values.astype(np.int16)+1)

mse = mean_squared_error(y, predictions)
print("\nMSE: ", round(mse, 2))





#%%
