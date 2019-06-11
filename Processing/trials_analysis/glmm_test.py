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


# Controllers
load_file = True
inspect_data = False

# %%
# ? plotting func
def plotter(y, exog, predictions):
    cols = exog.columns[1:]
    f,axarr= plt.subplots(figsize=(16, 10), ncols=len(cols)+1)

    for ax,col in zip(axarr, cols):
        ax.plot(exog[col].values, y,'o',color='r', label = 'Obs', alpha=.25)
        # ax.hist(exog[col].values*y, density=True, color=[.4, .4, .4], alpha=.3, bins=4)
        ax.plot(exog[col].values, predictions,'o',color='g', label = 'Pred', alpha=.25)

        sns.regplot(exog[col].values, predictions, logistic=True, truncate=True, scatter=False, ax=ax)

        ax.legend()
        ax.set(title=col, ylabel="escape_arm", xlabel=col, yticks=[0,1], yticklabels=["left", "right"], ylim=[-.1, 1.1])

    trials.head()

# %%
############################################################################################################################################################################################################################################################
##########################################                             # ! Get data
############################################################################################################################################################################################################################################################

# %%
# ! Load the data
file_path = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\saved_dataframes\\glm_data.pkl"

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
    trials["origin_arm_clean"] = ["right" if "Right" in arm else "left" for arm in trials['origin_arm'].values]
    trials["escape_arm_clean"] = ["right" if "Right" in arm else "left" for arm in trials['escape_arm'].values]

    trials = pd.get_dummies(trials, columns=["escape_arm_clean", "origin_arm_clean", "grouped_experiment_name"], 
                                    prefix=["escape", "origin", "experiment"])
else:
    trials = pd.read_pickle(file_path)

#%%
# ! Get the maze data

# ? get the maze params
parfile = "Processing/trials_analysis/params.yml"
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
# TODO merge the two to use maze stuff as predictors



# %% 
# ! Inspect data
if inspect_data:
    pd.plotting.scatter_matrix(trials, figsize=(16,12), s=1000)

    right_trials = trials.loc[trials.escape_arm == "Right_Medium"]
    left_trials = trials.loc[(trials.escape_arm == "Left_Medium") | (trials.escape_arm == "Left_Far")]
    asym_trials = trials.loc[trials.grouped_experiment_name == "asymmetric"]
    sym_trials = trials.loc[trials.grouped_experiment_name == "symmetric"]

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
############################################################################################################################################################################################################################################################
##########################################                              # ! GLM
############################################################################################################################################################################################################################################################
# %%
# ! GLM with stats model

# standardise the variables  and get the data
scaler = Scaler()

endog = trials.escape_right # ? to predict
exog = trials[["escape_duration", "mean_speed", "time_out_of_t"]] #, "mean_speed", "total_angular_displacement"]]
for c in exog.columns:
    if "arm" not in c:
        exog[c] = scaler.fit_transform(exog[c].values.reshape(-1, 1))
exog = sm.add_constant(exog) # add for intercept

# Define and fit the model
gamma_model = sm.GLM(endog, exog, family=sm.families.Binomial(link=sm.families.links.logit))
res = gamma_model.fit()
print(res.summary())

# ? plot the fitted model's results
y = endog.ravel()
sort_idx = np.argsort(y)
nobs = res.nobs
predictions = gamma_model.predict(res.params)

plotter(y, exog, predictions)
# %%
############################################################################################################################################################################################################################################################
##########################################                              # ! Logistic regression
############################################################################################################################################################################################################################################################
# %%

# ! try multivariate logistic regression
model = smf.logit(formula="escape_right ~ mean_speed + escape_duration + time_out_of_t", data=trials).fit()
print(model.summary())

predictions = model.predict()

plotter(y, trials[["escape_arm", "escape_duration", "time_out_of_t", "mean_speed"]], predictions)







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
