# %%
from Utilities.imports import *
import statsmodels.api as sm
from pandas.plotting import scatter_matrix

from Analysis.Behaviour.utils.T_utils import get_T_data, get_angles, get_above_yth
%matplotlib inline

# Getting data
aligned_trials  = get_T_data(load=True, median_filter=False)
# %%
n_trials = 125
n_frames = 10

ids = random.choices(np.arange(len(aligned_trials)), k=n_trials) 
outcomes = [1 if 'right' in e else 0 for e in list(aligned_trials.iloc[ids].escape_side.values)]

data = dict(x=[],
            y  = [],
            orientation = [],
            direction_of_mvmt = [],
            speed = [],
            # ang_speed = [],
            )

frames_outcomes = []
for trial_n, i in enumerate(ids):
    trial = aligned_trials.iloc[i]
    frames = np.random.randint(0, len(trial.tracking), n_frames)

    data['x'].extend(list(trial.tracking[frames, 0]))
    data['y'].extend(list(trial.tracking[frames, 1]))
    data['orientation'].extend(list(trial.body_orientation[frames]))
    data['direction_of_mvmt'].extend(list(trial.direction_of_movement[frames]))
    data['speed'].extend(list(trial.tracking[frames, 2]))
    # data['ang_speed'].extend(list(trial.body_angvel[frames]))

    frames_outcomes.extend([outcomes[trial_n] for n in range(n_frames)])

data['outcomes'] = frames_outcomes
data = pd.DataFrame(data)

left_trials, right_trials = data.loc[data.outcomes == 0], data.loc[data.outcomes == 1]


# %% 
# ! EXPLORE THE DATA
# Plot tracking (to find errors)
f, ax = create_figure(subplots=False, figsize=(16, 16), facecolor=white)
for n, i in enumerate(ids):
    scatter = ax.scatter(aligned_trials.iloc[i].tracking[:, 0], aligned_trials.iloc[i].tracking[:,1], 
                c=aligned_trials.iloc[i].body_orientation+90, cmap="bwr", vmin=0, vmax=360)

cm = f.colorbar(scatter)

# %%
# Pot hists
f, ax = create_figure(subplots=False, figsize=(16, 16), facecolor=white)
ax = data.hist(color=orange, ax=ax)


f, ax = create_figure(subplots=False, figsize=(16, 16), facecolor=white)
ax = left_trials.hist(color=blue, ax=ax)


f, ax = create_figure(subplots=False, figsize=(16, 16), facecolor=white)
ax = right_trials.hist(color=red, ax=ax)

# %%

# Plot scatter mtx
for d, ttl, color in zip([data, left_trials, right_trials],  ["all", "left", "right"], [orange, blue, red]):
    d = d.drop(['outcomes'], axis=1)
    f, ax = create_figure(subplots=False, figsize=(16, 16))
    _ = scatter_matrix(d, ax=ax, alpha=0.8, color=color, hist_kwds=dict(color=color, facecolor=color))
    ax.set(title="{} trials".format(ttl), facecolor=[.2, .2, .2])
















# %%
logit_model=sm.Logit(frames_outcomes,data)
result=logit_model.fit()
print(result.summary2())

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(data, frames_outcomes, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_prob_train = logreg.predict_proba(X_train)

y_pred = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
print("p(R) in test set was: {}".format(round(np.mean(y_test), 2)))
# %%
# plot results on train set
f, axarr = create_figure(subplots=True, ncols=2, facecolor=white, figsize=(16, 16))

axarr[0].scatter(X_train.x.values, X_train.y.values, c=y_train, cmap="bwr")

axarr[1].scatter(X_train.x.values, X_train.y.values, c=y_pred_prob_train[:, 1], cmap="bwr")


_ = axarr[0].set(title="TRAIN SET", facecolor=[.2, .2, .2],)
_ = axarr[1].set(title="Prediction", facecolor=[.2, .2, .2],)

# %%
# plot results on test set
f, axarr = create_figure(subplots=True, ncols=2, facecolor=white, figsize=(16, 16))

axarr[0].scatter(X_test.x.values, X_test.y.values, c=y_test, cmap="bwr")

axarr[1].scatter(X_test.x.values, X_test.y.values, c=y_pred_prob[:, 1], cmap="bwr")


_ = axarr[0].set(title="TEST SET", facecolor=[.2, .2, .2],)
_ = axarr[1].set(title="Prediction", facecolor=[.2, .2, .2],)

# %%
