# %%
from Utilities.imports import *
import statsmodels.api as sm

from Analysis.Behaviour.utils.T_utils import get_T_data, get_angles, get_above_yth
%matplotlib inline

# Getting data
aligned_trials  = get_T_data(load=True, median_filter=False)
# %%
n_trials = 100
n_frames = 100

ids = random.choices(np.arange(len(aligned_trials)), k=10) 
outcomes = [1 if 'right' in e else 0 for e in list(aligned_trials.iloc[ids].escape_side.values)]

data = dict(x=[],
            y  = [],
            orientation = [],
            direction_of_mvmt = [],
            speed = [],
            ang_speed = [],
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
    data['ang_speed'].extend(list(trial.body_angvel[frames]))

    frames_outcomes.extend([outcomes[trial_n] for n in range(n_frames)])

data = pd.DataFrame(data)

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

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
print("p(R) in test set was: {}".format(round(np.mean(y_test), 2)))
# %%
