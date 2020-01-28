# %%
import sys
sys.path.append('C:\\Users\\Federico\\Documents\\GitHub\\BehaviourAnalysis')
from Utilities.imports import *
import statsmodels.api as sm
from pandas.plotting import scatter_matrix

from scipy.optimize import curve_fit
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, mean_squared_error

from Analysis.Behaviour.utils.experiments_analyser import ExperimentsAnalyser
from Processing.rois_toolbox.rois_stats import convert_roi_id_to_tag

def save_plot(name, f):
    if sys.platform == 'darwin': 
        fld = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/plots/choice_summary"
    else:
        fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\plots\\choice_summary"
    f.savefig(os.path.join(fld,"svg", "{}.svg".format(name)))
    f.savefig(os.path.join(fld, "{}.png".format(name)))

%matplotlib inline


# %%
# Define a bunch of useful colors
palette = makePalette(green, orange, 7 , False)
maze_colors = {
    'm0': darkgreen,
    'm1': palette[0],
    'm1-dark': darkred, 
    'm1-light': red, 
    'm2': palette[1],
    'm3': palette[2],
    'm4': palette[3],
    'm6': salmon,
    'mb': palette[4],
    'mb1': palette[4],
    'mb2': palette[5]
}

palette = makePalette(teal, lilla, 4, False)
arms_colors = {
    "left": teal,
    'center': darkcyan,
    "right": lilla,
}


psychometric_mazes = ["m1", "m2", "m3", "m4"]
psychometric_mazes_and_dark = ["m1", "m2", "m3", "m4", "m1-dark"]
five_mazes = ["m1", "m2", "m3", "m4", "m6"]
m6 = ["m6"]
m0 = ["m0"]
allmazes = ["m1", "m2", "m3", "m4", "m6", "mb"]
arms = ['left', 'right', 'center']

# %%
# Getting data
ea = ExperimentsAnalyser(load_psychometric=False, tracking="all")
ea.max_duration_th = 9
# ea.add_condition("m0", maze_design=0, lights=1, escapes_dur=True, tracking="all"); print("Got m0")
ea.add_condition("m1", maze_design=1, lights=1, escapes_dur=True, tracking="all"); print("Got m1")
ea.add_condition("m2", maze_design=2, lights=1, escapes_dur=True, tracking="all"); print("Got m2")
ea.add_condition("m3", maze_design=3, lights=1, escapes_dur=True, tracking="all"); print("Got m3")
ea.add_condition("m4", maze_design=4, lights=1, escapes_dur=True, tracking="all"); print("Got m4")
ea.add_condition("m6", maze_design=6, lights=1, escapes_dur=True, tracking="all"); print("Got m6")

# ---------------------------------- cleanup --------------------------------- #
goodids, skipped = [], 0
trials = ea.conditions['m1']
for i, trial in trials.iterrows():
    if trial.escape_arm == "left":
        if np.max(trial.body_xy[:, 0]) > 600:
            skipped += 1
            continue
    goodids.append(trial.stimulus_uid)

t = ea.conditions['m1'].loc[ea.conditions['m1'].stimulus_uid.isin(goodids)]
print(len(t.loc[t.escape_arm == "right"])/len(t), len(trials.loc[trials.escape_arm == "right"])/len(trials))
ea.conditions['m1'] = t


for condition, trials in ea.conditions.items():
    print("Maze {} -- {} trials".format(condition, len(trials)))



# %%

# -------------------------------- PATH LENGTH ------------------------------- #

add_m0 = False
mazes = load_yaml("C:\\Users\\Federico\\Documents\\GitHub\\BehaviourAnalysis\\database\\maze_components\\Mazes_metadata.yml")

print("Extracting path length")
for maze, metadata in mazes.items():
    if maze not in five_mazes: continue
    print(" {} - ratio: {}".format(maze, round(metadata['left_path_length']/metadata['right_path_length'], 2)))
    mazes[maze]['ratio'] = metadata['left_path_length']/metadata['right_path_length']


# ---------------------------- EUCLIDEAN DISTANCE ---------------------------- #
print("Extracting euclidean distance")
euclidean_dists = {}
for i, (condition, trials) in enumerate(ea.conditions.items()):
    # Get data
    if condition not in five_mazes: continue

    means, maxes = {a:[] for a in ['left', 'right']}, {a:[] for a in ['left', 'right']}
    for n, trial in trials.iterrows():
        if trial.escape_arm == "center": continue

        d = calc_distance_from_shelter(trial.body_xy[trial.out_of_t_frame-trial.stim_frame:, :], [500, 850])
        means[trial.escape_arm].append(np.mean(d))
        maxes[trial.escape_arm].append(np.max(d))

    # Take average and save it
    y = [np.mean(means['left']), np.mean(means['right'])]
    euclidean_dists[condition] = y[0]/y[1]
    print(" {} - ratio: {}".format(maze, round(y[0]/y[1], 2)))


# ------------------------------ ESCAPE DURATION ----------------------------- #
print("Extracting escape duration")
path_durations, alldurations = ea.get_duration_per_arm_from_trials()

for i, (condition, durations) in enumerate(path_durations.items()):
    if condition not in five_mazes: continue
    y = [durations.left.mean, durations.right.mean]
    print(" {} - ratio: {}".format(condition, round(y[0]/y[1], 2)))


# %%
# ----------------------------- PREP DATA FOR GLM ---------------------------- #
# Get all trials
all_trials = dict(maze=[], geodist=[], eucldist=[], outcome=[], origin=[])
summary = dict(maze=[], geodist=[], eucldist=[], n=[], k=[], m=[], pr=[])

for condition, trials in ea.conditions.items():
    n,k = 0, 0
    for i, trial in trials.iterrows():
        if trial.escape_arm == "center": continue
        elif trial.escape_arm == "right": 
            all_trials['outcome'].append(1)
            k +=1
        else:
            all_trials['outcome'].append(0)
        n += 1

        if trial.origin_arm == 'right':
            all_trials['origin'].append(1)
        else: 
            all_trials['origin'].append(0)

        all_trials['geodist'].append(mazes[condition]['ratio'])
        all_trials['eucldist'].append(euclidean_dists[condition])
        all_trials['maze'].append(condition)

    summary['maze'].append(condition)
    summary['geodist'].append(mazes[condition]['ratio'])
    summary['eucldist'].append(euclidean_dists[condition])
    summary['k'].append(k)
    summary['n'].append(n)
    summary['m'].append(n-k)
    summary['pr'].append(k/n)


all_trials= pd.DataFrame(all_trials)
summary= pd.DataFrame(summary)
ntrials = len(all_trials)

summary

# %%
# -------------------------------- GLM -------------------------------- #
from statsmodels.graphics.api import abline_plot

params_combinations = [['eucldist'], ['geodist'], ['geodist', 'eucldist']]
models = []

for params in params_combinations:

    exog = summary[params]
    exog = sm.add_constant(exog, prepend=False)
    endog = summary[['k', 'm']]

    glm_binom = sm.GLM(endog, exog, family=sm.families.Binomial())
    res = glm_binom.fit()

    print("\n\n\Fitting GLM with parameters: {}\n".format(params))
    print(res.summary())
    models.append(res)
# print('\nParameters: \n', res.params)
# print('\nT-values: \n', res.tvalues)



# %%
# ------------------------------ EVALUATE MODELS ----------------------------- #

def compute(params, geo=1, eucl=1):
    x = params['const'] + params['geodist']*geo + params['eucldist']*eucl
    return 1/(1+np.exp(-x))


models_colors = [lightseagreen, blackboard, plum]
markers = ['*', 'o', 'v']
models_names = ['eucl', 'geod', 'eucl+geod']
# Plot y vs \hat{y} for each model

f, ax = create_figure(subplots=False)

for model, color, mk, name in zip(models, models_colors, markers, models_names):
    # Predict yhat
    nobs = model.nobs
    y = endog['k']/endog.sum(1)
    yhat = model.mu

    ax.scatter(yhat, y, s=350, marker = mk, color=color, zorder=99, label=name)

    print('Model {} --  m.s.e.: {}'.format(name, round(mean_squared_error(y, yhat), 4)))

            
ax.legend()
ax.plot([0, 1], [0, 1], ls="--", lw=2, color=black, alpha=.5)
ax.set(title="Model evaluation", xlabel='predicted p(R)', ylabel="real p(R)" )



# %%
# More plots
f, ax = create_figure(subplots=False)
ax.scatter(all_trials.geodist + np.random.normal(0, .02, size=ntrials), 
            all_trials.eucldist + np.random.normal(0, .02, size=ntrials), 
            cmap="bwr", c=all_trials.outcome, vmin=0, vmax=1, s=100, alpha=.25)
ax.scatter([mazes[condition]['ratio'] for condition in ea.conditions.keys()], 
                [euclidean_dists[condition] for condition in ea.conditions.keys()], 
                s=250, c=[maze_colors[condition] for condition in ea.conditions.keys()], edgecolor=black,  zorder=99)

for model, color, mk, name in zip(models, models_colors, markers, models_names):
    g = np.linspace(0, 3, num=250)
    if 'geodist' in model.params.keys() and 'eucldist' in model.params.keys():
        e = -(model.params['eucldist']/model.params['geodist'])*g - model.params['const']/model.params['geodist']
        ax.plot(g, e, color=color, lw=2, ls="--", label=name)
    elif 'geodist' in model.params.keys():
        ax.axvline(- model.params['const']/model.params['geodist'],  color=color, lw=2, ls="--", label=name)        
    else:
        ax.axhline(- model.params['const']/model.params['eucldist'],  color=color, lw=2, ls="--", label=name)        
ax.legend()
ax.set(xlim=[0.75, 2.3], ylim=[0.5, 1.3], xlabel='geodesic distance', ylabel='euclidean distance')



    # x0 = np.linspace(0.25, 3.5, num=250)
    # y = [compute(model.params, geo=x, eucl=1) for x in x0]
    # axarr[2].plot(x0, y, label="geodesic")

    # y = [compute(model.params, eucl=x, geo=1) for x in x0]
    # axarr[2].plot(x0, y, label="euclidean")

    # axarr[2].axhline(0.5,  ls="--", lw=2, color=black, alpha=.5)
    # axarr[2].legend()
    # axarr[2].set(title="logistic for each variabel", xlabel="distance ratio", xlim=[0, 3], ylabel="p(R)", ylim=[0, 1])


f.tight_layout()


# %%

# %%

# ----------------------------- SINGLE TRIAL GLM ----------------------------- #

from statsmodels.graphics.api import abline_plot


y, X = all_trials.outcome.values, all_trials[['geodist', 'eucldist']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

xtrain = pd.DataFrame(dict(geodist=X_train[:, 0], eucldist=X_train[:, 1]))
xtest = pd.DataFrame(dict(geodist=X_test[:, 0], eucldist=X_test[:, 1]))

params_combinations = [['geodist', 'eucldist']]
models = []

for params in params_combinations:

    exog = xtrain[params]
    exog = sm.add_constant(exog, prepend=False)
    endog = y_train

    glm_binom = sm.GLM(endog, exog, family=sm.families.Binomial())
    res = glm_binom.fit()

    print("\n\n\Fitting GLM with parameters: {}\n".format(params))
    print(res.summary())
    models.append(res)




# %%
def get_p(res, g, e):
    theta =  res.params['const'] + res.params['geodist']*g + res.params['eucldist']*e
    return 1/(1+np.exp(theta))

train_predicted_pr = [get_p(res, g, e) for e,g in zip(xtrain['eucldist'], xtrain['geodist'])]
test_predicted_pr = [get_p(res, g, e) for e,g in zip(xtest['eucldist'], xtest['geodist'])]

train_errors, errors = [], []
train_pcorr, test_pcorr = [], []
for i in range(1000):
    y_hat = np.random.binomial(1, test_predicted_pr)
    errors.append(mean_squared_error(y_test, y_hat))
    test_pcorr.append(np.where(y_hat == y_test)[0].shape[0]/len(y_test))

    y_hat = np.random.binomial(1, train_predicted_pr)
    train_errors.append(mean_squared_error(y_train, y_hat))
    train_pcorr.append(np.where(y_hat == y_train)[0].shape[0]/len(y_train))

print("Train m.s.e. {0:.4f} +- {1:.4f}".format(np.mean(train_errors), np.std(train_errors)))    
print("Test m.s.e. {0:.4f} +- {1:.4f}".format(np.mean(errors), np.std(errors)))    

print("Train p corr {0:.4f} +- {1:.4f}".format(np.mean(train_pcorr), np.std(train_pcorr)))    
print("Test p corr {0:.4f} +- {1:.4f}".format(np.mean(test_pcorr), np.std(test_pcorr)))    

# %%
# ------------------------------ EVALUATE MODEL ------------------------------ #

# Compute p(R) for each trial in the test set
exog = xtest[params]
exog = sm.add_constant(exog, prepend=False)
yhat = res.predict(exog).values

# Plot predicted p(R) with actual trial outcome + logistic regression
sort_idx = np.argsort(yhat)



f, ax = create_figure(subplots=False)
ax.plot(yhat[sort_idx], color=blackboard, lw=5, zorder=99, label='prediction')
ax = sns.regplot(np.arange(len(yhat)), y_test[sort_idx], ax=ax, logistic=True, n_boot=500, y_jitter=.03, label='data fit',
            scatter_kws = dict(color=darksalmon, s=25, alpha=.6),
            line_kws = dict(color=darksalmon, alpha=.8), truncate=True,)
_ = ax.set(xlabel='trials sorted by predicted p(R)', ylabel='p(R)', title='performance on TEST SET')
_ = ax.legend()
# %%
