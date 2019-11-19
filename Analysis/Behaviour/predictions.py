# %%
import sys
sys.path.append('./')
from Utilities.imports import *
import statsmodels.api as sm
from pandas.plotting import scatter_matrix

from Analysis.Behaviour.utils.experiments_analyser import ExperimentsAnalyser
# %matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics#
# %%
# Getting data
ea = ExperimentsAnalyser(load_psychometric=False, tracking="threat")
ea.add_condition("m2", maze_design=2, escapes_dur=True, tracking="threat")
ea.add_condition("m3", maze_design=3, escapes_dur=True, tracking="threat")


trials = ea.merge_conditions_trials(list(ea.conditions.values()))

print("Found {} trials".format(len(trials)))
print("\n", trials.head())
# Explore the data
print("\np(R) : {}".format(round(list(trials.escape_arm).count("right")/len(trials), 2)))


# %%
# useful funcs and params
n_frames = 20
n_trials_per_side = 80

def get_dataframe(miny, maxy):
    data = dict(
        x=[], y=[], dir_mvmt=[],  orientation=[], escape_arm=[]) # speed=[],

    n_left, n_right = 0, 0
    for trial_n, trial in trials.iterrows():
        # get y only below a threshold
        y = trial.body_xy[:, 1]
        good_idx = np.where((y>miny)&(y<maxy))
        y = y[good_idx]
        if not np.any(y): continue

        # get random frame numbers
        frames = np.random.randint(0, len(y), n_frames)

        # Get equal number per side
        side = trial.escape_arm
        if side == "left" and n_left < n_trials_per_side:
            n_left += 1
        elif side == "left" and n_left > n_trials_per_side:
            continue
        elif side == "right" and n_right < n_trials_per_side:
            n_right += 1
        else: 
            continue

        # prep stuff
        x = np.nan_to_num(trial.body_xy[:, 0][good_idx])
        o = np.nan_to_num(trial.body_orientation[good_idx])
        dm = np.nan_to_num(trial.body_dir_mvmt[good_idx])
        s = np.nan_to_num(trial.body_speed[good_idx])


        # put everything together
        data['x'].extend(list(x[frames]))
        data['y'].extend(list(np.nan_to_num(y[frames])))
        data['orientation'].extend(list(o[frames]))
        data['dir_mvmt'].extend(list(dm[frames]))
        # data['speed'].extend(list(s[frames]))
        data['escape_arm'].extend(list(np.nan_to_num([1 if trial.escape_arm == "right" else 0 for i in range(len(frames))])))
        

    data = pd.DataFrame(data)

    # Fix the orientation column
    oris = data['orientation'].values
    oris -= 180
    oris[oris<0] += 360

    data['orientation'] = oris
    # print(data.head())

    frames_outcomes = data.escape_arm.values
    data = data.drop(columns=["escape_arm"])

    # print("\n{} frame in total from {} trials".format(len(data), n_left+n_right))

    return data, frames_outcomes


def run_test_logistic(data, frames_outcomes):
    X_train, X_test, y_train, y_test = train_test_split(data, frames_outcomes, test_size=0.3, random_state=0)
    logreg = LogisticRegression(verbose=True)
    logreg.fit(X_train, y_train)
    y_pred_prob_train = logreg.predict_proba(X_train)

    y_pred = logreg.predict(X_test)
    y_pred_prob = logreg.predict_proba(X_test)

    accuracy = logreg.score(X_test, y_test)
    real_pr = round(np.mean(y_test), 2)
    
    return logreg, (X_train, y_train), (X_test, y_test), (y_pred_prob_train, y_pred_prob), real_pr, accuracy


# %%
# Analyze by slice
slices = [(150, 170), (180, 200), (210, 230), (240, 260), (270, 290), (300, 320), (330, 350)]

# slices = [(0, 500)]
f, axarr = create_figure(subplots=True, ncols=4, facecolor=white, figsize=(16, 16))


cols = ["x", "y"]
coeffs = {c:[] for c in cols}
ypos, xval = [], []
for miny, maxy in slices:
    data, frames_outcomes = get_dataframe(miny, maxy)

    # keep columns
    data = data[cols]

    # do logistic regression
    logreg, trainingset, testset, predictions, pr, predicted_pr =  run_test_logistic(data, frames_outcomes)

    # plot frames
    axarr[0].scatter(trainingset[0].x.values, trainingset[0].y.values, c=trainingset[1], cmap="bwr", alpha=.5)
    axarr[1].scatter(trainingset[0].x.values, trainingset[0].y.values, c=predictions[0][:, 1], cmap="bwr", alpha=.5)

    # store values for other plots
    ypos.append(round(np.mean([miny, maxy])))
    xval.append((pr, predicted_pr))

    for col, coef in zip(cols, logreg.coef_):
        coeffs[col].append(cofff)


    ## speed:          {}
#     print(""" Instructions
#     Logistic regression parameters:
#         x:              {}
#         y:              {}
#         dir_mvmt:       {}
        
#         orientation:    {}

# """.format(*[round(x,5) for x in logreg.coef_[0]]))

# plot p(R) and accuracy
axarr[2].barh([y+5 for y in ypos], [x[0] for x in xval], align='center', color=green, label="p(R)", height=10)
axarr[2].barh([y-5 for y in ypos], [x[1] for x in xval], align='center', color=orange, label="accuracy", height=10)
axarr[2].legend()

# plot p(R) and accuracy 
# TODO finishsshs
# !sdoifhsduiogfnsdognfiosdfskl
for cname, coeff in ceffs.items():
    axarr[3].barh([y for y in ypos], [x[0] for x in xval], align='center', color=green, label="p(R)", height=10)
axarr[3].legend()

# set axes props
_ = axarr[0].set(title="TRAIN SET", facecolor=[.2, .2, .2], xlim=[420, 580])
_ = axarr[1].set(title="Prediction", facecolor=[.2, .2, .2], xlim=[420, 580])
axarr[2].set(title="p(R) vs performance", facecolor=[.2, .2, .2], xlim=[0, 1.05])

# %%
# explore
scatter_matrix(data, alpha=0.2, figsize=(20, 20), diagonal='kde')
plt.show()


# %%
logit_model=sm.Logit(frames_outcomes,data)
result=logit_model.fit()
print(result.summary2())

# %%


# plot results on train set


# %%
# plot results on test set
f, axarr = create_figure(subplots=True, ncols=2, facecolor=white, figsize=(16, 16))

axarr[0].scatter(X_test.x.values, X_test.y.values, c=y_test, cmap="bwr")

axarr[1].scatter(X_test.x.values, X_test.y.values, c=y_pred_prob[:, 1], cmap="bwr")


_ = axarr[0].set(title="TEST SET", facecolor=[.2, .2, .2],)
_ = axarr[1].set(title="Prediction", facecolor=[.2, .2, .2],)

# %%
