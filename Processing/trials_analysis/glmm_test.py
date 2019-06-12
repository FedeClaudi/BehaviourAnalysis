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

# Controllers
load_file = True
inspect_data = False

logistic_th = .7

# %%
# ? plotting func
def plotter(y, exog, predictions, plot=True):
    cols = exog.columns[1:]
    if plot: f,axarr= plt.subplots(figsize=(16, 10), ncols=len(cols)+1)

    bin_prediction = np.zeros_like(predictions) + .1
    bin_prediction[predictions > logistic_th] = .9

    correct_bin = np.zeros_like(predictions) + .4
    correct_bin[(predictions > logistic_th) & (y > logistic_th)] = 0.6

    if plot:
        for ax,col in zip(axarr, cols):
            ax.plot(exog[col].values, y,'o',color='r', label = 'Obs', alpha=.25)

            ax.plot(exog[col].values, predictions,'o',color='g', label = 'Pred', alpha=.25)
            ax.plot(exog[col].values, bin_prediction,'o',color='b', label = 'Pred - binary', alpha=.25)
            ax.plot(exog[col].values, correct_bin,'o',color='k', label = 'Correctly pred', alpha=.25)

            ax.axhline(logistic_th, color="k")

            sns.regplot(exog[col].values, predictions, logistic=True, truncate=True, scatter=False, ax=ax)

            ax.legend()
            ax.set(title=col, ylabel="escape_arm", xlabel=col, yticks=[0,1], yticklabels=["left", "right"], ylim=[-.1, 1.1])

    axarr[-1].plot(y,'o',color='r', label = 'Obs', alpha=.25)
    axarr[-1].plot(predictions,'o',color='g', label = 'Pred', alpha=.25)

    n_corr = (correct_bin > logistic_th).sum()
    return "Correctly predicted: {} of {} - {}%".format(n_corr, len(correct_bin), round(n_corr/len(correct_bin)*100, 2))

def plotter2(y, predictions):
    x = np.arange(len(predictions))
    # fit = np.poly1d(np.polyfit(x, predictions, 2))
    # fitted = [fit(xx) for xx in x]

    sort_idxs = np.argsort(y)

    f, ax = plt.subplots(figsize=(9, 8),)
    ax.plot(y[sort_idxs],'o',color='r', label = 'Obs', alpha=.25)
    ax.plot(predictions[sort_idxs],'o',color='g', label = 'Pred', alpha=.25)
    # sns.regplot(x, predictions[sort_idxs], logistic=True, 
    #                             truncate=True, scatter=False, ax=ax)
    ax.set(ylabel="escape_arm", xlabel="trials") #, yticks=[0,1], yticklabels=["left", "right"])



# %%
############################################################################################################################################################################################################################################################
##########################################                             # ! Get data
############################################################################################################################################################################################################################################################

# %%
# ! Load the data
if sys.platform != "darwin":
    file_path = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\saved_dataframes\\glm_data.pkl"
    parfile = None
else:
    parfile = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/analysis_metadata/maze/escape_paths_features.yml"
    file_path = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/analysis_metadata/saved_dataframes/glm_data.pkl"


#%%
if not load_file:
    # load and clean
    trial_data = Trials(exp_1_mode=True)
    trials = trial_data.trials.drop(["is_escape", "experiment_name", "outward_tracking_data"], axis=1)

    # Get some extra metrics
    mean_escape_speed = []
    max_escape_speed = [] # ? actually 95th perc to remove outlisers 
    escape_path_len = []
    total_angular_displacement = []

    for i, trial in trials.iterrows():
        escape_path_len.append(trial.tracking_data.shape[0]/trial.fps)
        mean_escape_speed.append(np.mean(trial.tracking_data[:, 2]))
        max_escape_speed.append(np.percentile(trial.tracking_data[:, 2], 95))

        angles = calc_angle_between_points_of_vector(trial.tracking_data[:, :2])
        total_angular_displacement.append(np.sum(np.abs(calc_ang_velocity(angles))))

    trials['mean_speed'] = mean_escape_speed
    trials['max_speed'] = max_escape_speed
    trials['escape_duration'] = escape_path_len
    trials['total_angular_displacement'] = total_angular_displacement

    # Clean up
    # Make ToT a float
    trials = trials.loc[trials['time_out_of_t'] > 0]
    trials["time_out_of_t"] = np.array(trials['time_out_of_t'].values, np.float64)

    # Fix arms categoricals
    # TODO check origin
    trials["origin_arm_clean"] = ["right" if "Right" in arm else "left" for arm in trials['origin_arm'].values]
    trials["escape_arm_clean"] = ["right" if "Right" in arm else "left" for arm in trials['escape_arm'].values]

    trials = pd.get_dummies(trials, columns=["escape_arm_clean", "origin_arm_clean", "grouped_experiment_name"], 
                                    prefix=["escape", "origin", "experiment"])
else:
    trials = pd.read_pickle(file_path)
    trials = trials.drop("origin_left", axis=1)
    trials["origin_arm_clean"] = ["right" if "Right" in arm[0] else "left" for arm in trials['origin_arm'].values]
    trials = pd.get_dummies(trials, columns=["origin_arm_clean"], prefix=["origin"])

#%%
# ? get the maze params
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


# %% 
# ? merge the two datasets
agent_len, theta_start, theta_tot = [], [], []
for i, trial in trials.iterrows():
    if trial.experiment_asymmetric:
        if "Left" in trial.escape_arm:
            arm_data = params.iloc[0]
        else:
            arm_data = params.iloc[1]
    else:
        if "Left" in trial.escape_arm:
            arm_data = params.iloc[6]
        else:
            arm_data = params.iloc[7]

    agent_len.append(arm_data.rLen)
    theta_start.append(arm_data.theta_start)
    theta_tot.append(arm_data.theta_tot)

trials["agent_len"] = agent_len
trials["theta_start"] = theta_start
trials["theta_tot"] = theta_tot


# %% 
# ! Inspect data
if inspect_data:
    pd.plotting.scatter_matrix(trials, figsize=(16,12), s=1000)

    right_trials = trials.loc[trials.escape_arm == "Right_Medium"]
    left_trials = trials.loc[(trials.escape_arm == "Left_Medium") | (trials.escape_arm == "Left_Far")]
    asym_trials = trials.loc[trials.experiment_asymmetric == 1]
    sym_trials = trials.loc[trials.experiment_asymmetric == 0]

    f, axarr = plt.subplots(ncols=3)

    right_line, right_scatter = {"color":"green", "linewidth":2,  "label":"ASYM"}, {"color":"green", "alpha":.2, "s":150}
    left_line, left_scatter = {"color":"red", "linewidth":2,  "label":"SYM"}, {"color":"red", "alpha":.2, "s":150}

    keywords =[["mean_speed", "escape_duration"], ["mean_speed", "time_out_of_t"], ["escape_duration", "time_out_of_t"]]
    titles = ["Duration vs mean speed", " time_out_of_t vs mean speed", "escape dur vs ToT"]

    for i, ((k1, k2), ax, title) in enumerate(zip(keywords, axarr, titles)):
        if i == 2: o = 1
        else: o = 3
        sns.regplot(asym_trials[k1], asym_trials[k2], ax=ax, 
                                order=o, ci=95, truncate=True, line_kws=right_line,  scatter_kws=right_scatter)
        sns.regplot(sym_trials[k1], sym_trials[k2], ax=ax,
                                order=o, ci=95, truncate=True, line_kws=left_line,  scatter_kws=left_scatter)

        ax.set(title=title)
        ax.legend()


# %%
# ! drop part of the data
# trials = trials.loc[trials.experiment_asymmetric == 1]

# Split training and test

train, test = train_test_split(trials, test_size=0.2)

# ? apply transforms
test["log_escape_duration"] = np.log(test["escape_duration"])


# %%
############################################################################################################################################################################################################################################################
##########################################                              # ! GLM
############################################################################################################################################################################################################################################################
# %%
# ! GLM with stats model
# TODO add position at trials onset to predicting vars
scaler = Scaler()
# TODO look at p(R) for HIGH and LOW ToT trials
exog = test[["log_escape_duration", ]]
for c in exog.columns:
    # exog[c] = scaler.fit_transform(exog[c].values.reshape(-1, 1))
    exog[c] = exog[c].values.reshape(-1, 1)

exog = sm.add_constant(exog)

# Define and fit the model
# gamma_model = sm.GLM(endog, exog, family=sm.families.Binomial(link=sm.families.links.logit))

gamma_model = smf.glm(formula = "time_out_of_t ~  np.log(escape_duration) ", data=train,
                    family=sm.families.Gamma(link=sm.families.links.log))
res = gamma_model.fit_regularized()
print(res.params)

# ? plot the fitted model's results
y = test.time_out_of_t.ravel()
predictions = gamma_model.predict(res.params, exog)
plotter2(y,  predictions)

mse = mean_squared_error(y, predictions)
print("\nMSE: ", round(mse, 2))

# %%
############################################################################################################################################################################################################################################################
##########################################                              # ! Logistic regression
############################################################################################################################################################################################################################################################
# %%
"""
    If the model always predict Right as escape the % of corrects would be 68%
"""
# TODO normalise data
# ! try multivariate logistic regression
model = smf.logit(formula="escape_right ~  np.log(mean_speed) ", data=trials).fit()
print(model.summary())

predictions = model.predict()

plotter2(y, predictions, )

print(correct)

cols = trials.columns



# %%
############################################################################################################################################################################################################################################################
##########################################                              # ! GLMM
############################################################################################################################################################################################################################################################

# %%
 
# ! set up for the GLMM

formula = "mean_speed ~ np.multiply(escape_duration, escape_duration) + time_out_of_t" 
Y, X = dmatrices(formula, data=trials, return_type='matrix')
Terms  = X.design_info.column_names
_, Z   = dmatrices('mean_speed ~ -1+grouped_experiment_name', data=trials, return_type='matrix')

X      = np.asarray(X) # fixed effect
Y      = np.asarray(Y).flatten()
Z      = np.asarray(Z) # mixed effect

nfixed = np.shape(X)
nrandm = np.shape(Z)

print(Y.shape, X.shape)# , Z.shape)

#  PMC3 model
beta0     = np.linalg.lstsq(X,Y)  # Solves the equation a x = b by computing a vector x that minimizes the Euclidean 2-norm || b - a x ||^2. 

fixedpred = np.argmax(X,axis=1)
randmpred = np.argmax(Z,axis=1)


# %%
# ! Estimate parameters with PyMC3
with pm.Model() as glmm1:
    # Fixed effect
    beta = pm.Normal('beta', mu=0., sd=100., shape=(nfixed[1]))
    # random effect

    eps  = pm.HalfCauchy('eps', 5.)
    
    mu_est = pm.Deterministic('mu_est', tt.dot(X,beta)+tt.dot(Z,b))
    RT = pm.Normal('RT', mu_est, eps, observed=Y)
    
    trace = pm.sample(1000, tune=1000)
    
pm.traceplot(trace)


#%%
# ? check model

fixed_pymc = pm.summary(trace, varnames=['beta'])
b = np.asarray(fixed_pymc['mean'])
randm_pymc = pm.summary(trace, varnames=['b'])
q = np.asarray(randm_pymc['mean'])


f, ax = plt.subplots(figsize=(8,6))

sort_idx = np.argsort(Y)

ax.plot(Y[sort_idx],'o',color='r', label = 'Obs: Mean Speed', alpha=1)
fitted = np.dot(X,b).flatten()[sort_idx] # + np.dot(Z,q).flatten()
ax.plot(fitted, lw=3, label="Model Mean Speed", alpha=.8)

ax.legend()
