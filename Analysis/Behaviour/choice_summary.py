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
from sklearn.metrics import confusion_matrix

from Analysis.Behaviour.utils.experiments_analyser import ExperimentsAnalyser
from Processing.rois_toolbox.rois_stats import convert_roi_id_to_tag

def save_plot(name, f):
    if sys.platform == 'darwin': 
        fld = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/plots/choice_summary"
    else:
        fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\plots\\choice_summary"
    f.savefig(os.path.join(fld,"svg", "{}.svg".format(name)))
    f.savefig(os.path.join(fld, "{}.png".format(name)))

# %matplotlib inline


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

# TODO write functino that gets N,n,M for each experiment
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
# ea.add_condition("m1-light", maze_design=1, lights=1, escapes_dur=True, tracking="all"); print("Got m1-light")
# ea.add_condition("m1-dark", maze_design=1, lights=0, escapes_dur=True, tracking="all"); print("Got m1-dark")


# ea.add_condition("twolong", maze_design=None, lights=None, escapes_dur=False, tracking="all", experiment_name="TwoArmsLong Maze"); print("Got TwoArmsLong Maze")
# ea.add_condition("ff", maze_design=None, lights=None, escapes_dur=True, tracking="all", experiment_name="FlipFlop Maze"); print("Got FlipFlop Maze")
# ea.add_condition("ff2", maze_design=None, lights=None, escapes_dur=True, tracking="all", experiment_name="FlipFlop2 Maze"); print("Got FlipFlop2 Maze")
# ea.add_condition("fourlong", maze_design=None, lights=None, escapes_dur=True, tracking="all", experiment_name="FourArms Maze"); print("Got FourArms Maze")
# ea.add_condition("mb", maze_design=None, lights=None, escapes_dur=True, tracking="all", experiment_name="Model Based"); print("Got Model Based")
# ea.add_condition("mb2", maze_design=None, lights=None, escapes_dur=True, tracking="all", experiment_name="Model Based V2"); print("Got Model Based V2")


for condition, trials in ea.conditions.items():
    print("Maze {} -- {} trials".format(condition, len(trials)))

# plot tracking
f, axarr = create_figure(subplots=True, ncols=int(np.ceil(len(ea.conditions.keys())/2)), nrows=2)
for n, (condition, trials) in enumerate(ea.conditions.items()):
    for i, trial in trials.iterrows():
        axarr[n].plot(trial.body_xy[:, 0], trial.body_xy[:, 1], color=arms_colors[trial.escape_arm])
    axarr[n].set(title=condition, xlim=[0, 1000], ylim=[0, 1000])
save_plot("tracking", f)



# %%
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


# %%
# --------------------------------------------------------------------------- #
#                                 ! PATH LENGTH                               #
# --------------------------------------------------------------------------- #
add_m0 = False

mazes = load_yaml("database/maze_components/Mazes_metadata.yml")
f, ax = create_figure(subplots=False)

ax.plot([0, 10000], [0, 10000], ls=':', lw=2, color=[.2, .2, .2], alpha=.3)

for maze, metadata in mazes.items():
    if maze == "m0": 
        if add_m0:
            for longer in ['right', 'left']:
                y = metadata['{}_path_length'.format(longer)]
                x = metadata['center_path_length']
                ax.scatter(x, y, color=maze_colors[maze], edgecolor=black, s=250, zorder=99)
                _ = vline_to_point(ax, x, y, color=maze_colors[maze], ymin=0, ls="--", alpha=.5, zorder=0)
                _ = hline_to_point(ax, x, y, color=maze_colors[maze], ls="--", alpha=.5, zorder=0)
    else:
        ax.scatter(metadata['right_path_length'], metadata['left_path_length'], color=maze_colors[maze], edgecolor=black, s=250, zorder=99)
        _ = vline_to_point(ax, metadata['right_path_length'], metadata['left_path_length'], color=maze_colors[maze], ymin=0, ls="--", alpha=.5, zorder=0)
        _ = hline_to_point(ax, metadata['right_path_length'], metadata['left_path_length'], color=maze_colors[maze], ls="--", alpha=.5, zorder=0)

    print("{} - ratio: {}".format(maze, round(metadata['left_path_length']/metadata['right_path_length'], 2)))
    mazes[maze]['ratio'] = metadata['left_path_length']/metadata['right_path_length']

_ = ax.set(title="Path lengths", xlabel='length of shortest', ylabel='length of longest', xlim=[400, 1150], ylim=[400, 1150])
save_plot("path_lengths", f)


# %%
# ---------------------------------------------------------------------------- #
#                             ! EUCLIDEAN DISTANCE                             #
# ---------------------------------------------------------------------------- #

plot_single_trials = True

euclidean_dists = {}
f, ax = create_figure(subplots=False)
xticks, xlabels = [], []
for i, (condition, trials) in enumerate(ea.conditions.items()):
    # Get data
    if condition not in five_mazes: continue

    means, maxes = {a:[] for a in ['left', 'right']}, {a:[] for a in ['left', 'right']}
    for n, trial in trials.iterrows():
        if trial.escape_arm == "center": continue

        d = calc_distance_from_shelter(trial.body_xy[trial.out_of_t_frame-trial.stim_frame:, :], [500, 850])
        means[trial.escape_arm].append(np.mean(d))
        maxes[trial.escape_arm].append(np.max(d))

    

    # Make plot
    x = [i-.25, i+.25]
    xticks.extend([i-.25,i, i+.25])
    xlabels.extend(["left","\n{}".format(condition), "right"])
    ax.axvline(i-.25, color=[.2, .2, .2], ls=":", alpha=.15)
    ax.axvline(i+.25, color=[.2, .2, .2], ls=":", alpha=.15)

    y = [np.mean(means['left']), np.mean(means['right'])]
    yerr = [stats.sem(means['left']), stats.sem(means['right'])]
    ax.plot(x, y, "-o", label=condition, color=maze_colors[condition], zorder=90)
    ax.scatter(x, y, edgecolor=black, s=250, color=maze_colors[condition], zorder=99)
    ax.errorbar(x, y, yerr, color=maze_colors[condition], zorder=90)


    ttest, pval = stats.ttest_ind(means['left'], means['right'])
    if pval < .05:
        ax.plot([i-.3, i+.3], [505, 505], lw=4, color=[.4, .4, .4])
        ax.text(i-0.025, 505, "*", fontsize=20)
    else:
        ax.plot([i-.3, i+.3], [505, 505], lw=4, color=[.7, .7, .7])
        ax.text(i-0.05, 508, "n.s.", fontsize=16)

    # Take average and save it
    euclidean_dists[condition] = y[0]/y[1]

_ = ax.set(title="Average euclidean distance", xticks=xticks, xticklabels=xlabels,
                ylabel="mean distance (s)")
save_plot("euclidean_dist", f)






# %%
# ---------------------------------------------------------------------------- #
#                               ! ESCAPE DURATION                              #
# ---------------------------------------------------------------------------- #

path_durations, alldurations = ea.get_duration_per_arm_from_trials()

xticks, xlabels = [], []
f, ax = create_figure(subplots=False)
for i, (condition, durations) in enumerate(path_durations.items()):
    if condition not in five_mazes: continue
    x = [i-.25, i+.25]
    xticks.extend([i-.25,i, i+.25])
    xlabels.extend(["left","\n{}".format(condition), "right"])
    y = [durations.left.mean, durations.right.mean]
    yerr = [durations.left.sem, durations.right.sem]

    print("Maze {} - ratio: {}".format(condition, round(y[0]/y[1], 2)))

    ax.axvline(i-.25, color=[.2, .2, .2], ls=":", alpha=.15)
    ax.axvline(i+.25, color=[.2, .2, .2], ls=":", alpha=.15)

    ax.plot(x, y, "-o", label=condition, color=maze_colors[condition], zorder=90)
    ax.scatter(x, y, edgecolor=black, s=250, color=maze_colors[condition], zorder=99)
    ax.errorbar(x, y, yerr, color=maze_colors[condition], zorder=90)

    ttest, pval = stats.ttest_ind(alldurations[condition]['left'], alldurations[condition]['right'])
    if pval < .05:
        ax.plot([i-.3, i+.3], [4.75, 4.75], lw=4, color=[.4, .4, .4])
        ax.text(i-0.025, 4.75, "*", fontsize=20)
    else:
        ax.plot([i-.3, i+.3], [4.75, 4.75], lw=4, color=[.7, .7, .7])
        ax.text(i-0.05, 4.8, "n.s.", fontsize=16)
        

_ = ax.set(title="Paths durations per maze design", xticks=xticks, xticklabels=xlabels,
                ylabel="mean duration (s)")
save_plot("escape_durations", f)





# %%
# ---------------------------- DURATION VS LENGTH ---------------------------- #
f, ax = create_figure(subplots=False)
durs, dists, speeds = [], [], []
for condition, trials in ea.conditions.items():
    if condition in ["m0", "m6"]: continue
    for n, (i, trial) in enumerate(trials.iterrows()):
        dist = np.sum(calc_distance_between_points_in_a_vector_2d(trial.body_xy[trial.out_of_t_frame-trial.stim_frame:,  :]))
        dur = (trial.at_shelter_frame - trial.out_of_t_frame)/trial.fps
        durs.append(dur); dists.append(dist); speeds.append(np.mean(trial.body_speed[trial.out_of_t_frame-trial.stim_frame:]))

ax.scatter(dists, durs, c=speeds, cmap="inferno_r", alpha=.7)
ax.set(xlabel="Distance", ylabel="Duration")
save_plot("dur_vs_dist", f)





# %%
# ---------------------------------------------------------------------------- #
#                             ! ALL TRIALS LOGISTIC                            #
# ---------------------------------------------------------------------------- #

# Get all trials
all_trials = dict(geodist=[], eucldist=[], outcome=[])
summary = dict(geodist=[], eucldist=[], n=[], k=[], m=[])

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

        all_trials['geodist'].append(mazes[condition]['ratio'])
        all_trials['eucldist'].append(euclidean_dists[condition])

    summary['geodist'].append(mazes[condition]['ratio'])
    summary['eucldist'].append(euclidean_dists[condition])
    summary['k'].append(k)
    summary['n'].append(n)
    summary['m'].append(n-k)


all_trials= pd.DataFrame(all_trials)
summary= pd.DataFrame(summary)
ntrials = len(all_trials)

# f, ax = create_figure(subplots=False)

# ax.scatter(all_trials.geodist + np.random.normal(0, .05, size=ntrials), 
#             all_trials.eucldist + np.random.normal(0, .05, size=ntrials), 
#             cmap="bwr", c=all_trials.outcome, vmin=0, vmax=1, s=25, alpha=.75)


y, X = all_trials.outcome.values, all_trials[['geodist', 'eucldist']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('\n\nAccuracy of logistic regression classifier on training set: {:.2f}, average p(R) in training set: {:.2f}\n'.format(logreg.score(X_train, y_train), np.mean(y_train)))

print('Accuracy of logistic regression classifier on test set: {:.2f}, average p(R) in test set: {:.2f}\n'.format(logreg.score(X_test, y_test), np.mean(y_test)))
cm = confusion_matrix(y_test, y_pred)
print(cm, "\n\n")


logit_model=sm.Logit(y,all_trials[['geodist', 'eucldist']])
result=logit_model.fit()
print(result.summary2())



# %%
# -------------------------------- TRYING GLM -------------------------------- #
from statsmodels.graphics.api import abline_plot

exog = summary[['geodist', 'eucldist']]
exog = sm.add_constant(exog, prepend=False)
endog = summary[['k', 'm']]

glm_binom = sm.GLM(endog, exog, family=sm.families.Binomial())
res = glm_binom.fit()
print(res.summary())
# print('\nParameters: \n', res.params)
# print('\nT-values: \n', res.tvalues)


nobs = res.nobs
y = endog['k']/endog.sum(1)
yhat = res.mu


f, axarr = create_figure(subplots=True, ncols=2, nrows=2)

axarr[0].scatter(yhat, y, s=250, c=darksalmon, zorder=99)
axarr[0].plot([0, 1], [0, 1], ls="--", lw=2, color=black, alpha=.5)
axarr[0].set(title="Model evaluation", xlabel='predicted p(R)', ylabel="real p(R)" )


axarr[1].scatter(all_trials.geodist + np.random.normal(0, .05, size=ntrials), 
            all_trials.eucldist + np.random.normal(0, .05, size=ntrials), 
            cmap="bwr", c=all_trials.outcome, vmin=0, vmax=1, s=100, alpha=.5)
axarr[1].scatter([mazes[condition]['ratio'] for condition in ea.conditions.keys()], 
                [euclidean_dists[condition] for condition in ea.conditions.keys()], 
                s=250, c=[maze_colors[condition] for condition in ea.conditions.keys()], edgecolor=black,  zorder=99)


x0 = np.linspace(0, 3, num=250)
x1 = [(-res.params['const']/res.params['geodist'] - (res.params['eucldist']/res.params['geodist'])*x) for x in x0]
axarr[1].plot(x0, x1, color=black)
axarr[1].set(xlim=[0.75, 2.3], ylim=[0.5, 1.3], xlabel='geodesic distance', ylabel='euclidean distance', title='all trials ')


def compute(params, geo=1, eucl=1):
    x = params['const'] + params['geodist']*geo + params['eucldist']*eucl
    return 1/(1+np.exp(-x))

x0 = np.linspace(0.25, 3.5, num=250)
y = [compute(res.params, geo=x, eucl=1) for x in x0]
axarr[2].plot(x0, y, label="geodesic")

y = [compute(res.params, eucl=x, geo=1) for x in x0]
axarr[2].plot(x0, y, label="euclidean")

axarr[2].axhline(0.5,  ls="--", lw=2, color=black, alpha=.5)
axarr[2].legend()
axarr[2].set(title="logistic for each variabel", xlabel="distance ratio", xlim=[0, 3], ylabel="p(R)", ylim=[0, 1])

# for m in psychometric_mazes:
#     axarr[2].scatter(mazes[m]['ratio'], pRs.loc[pRs.condition == m]['mean'], color=maze_colors[m], zorder=99)
# axarr[2].scatter(euclidean_dists['m6'], pRs.loc[pRs.condition == 'm6']['mean'], color=maze_colors['m6'], zorder=99)

f.tight_layout()



# %%
# ---------------------------------------------------------------------------- #
#                                ! PSYCHOMETRIC                                #
# ---------------------------------------------------------------------------- #

includem0 = False
includem6 = True
fit_curve = True

use_eucl = True
use_combined = True
use_duration = False

alpha, beta = .75, 1
if use_duration:
    combined_dists = {a:(alpha*path_durations[a].ratio.mean)+(beta*euclidean_dists[a]) for a in euclidean_dists.keys()}
else:
    combined_dists = {a:(alpha*mazes[a]['ratio'])+(beta*euclidean_dists[a]) for a in euclidean_dists.keys()}
if not use_combined:
    combined_dists = euclidean_dists

# Calc and plot pR for psychometric data
pRs = ea.bayes_by_condition_analytical()

f, ax = create_figure(subplots=False)

xfit, yfit, stdfit = [], [], []
X, Y, Xlabels= [], [], []
for i, pr in pRs.iterrows():
    if pr.condition not in psychometric_mazes: continue
    std = math.sqrt(pr.sigmasquared)
    yerr = np.array([pr['mean']-pr.prange.low, pr.prange.high-pr['mean']]).reshape(2, 1)
    if not use_eucl:
        if not use_duration:
            x = mazes[pr.condition]['ratio']
        else:
            x = path_durations[pr.condition].ratio.mean
    else:
        x = combined_dists[pr.condition]
    X.append(x)
    Y.append(pr['mean'])
    xfit.append(x); yfit.append(pr['mean']); stdfit.append(math.sqrt(pr['sigmasquared']))
    color=maze_colors[pr.condition]
    Xlabels.append("{}\n{}".format(pr.condition, round(x,1)))

    ax.errorbar(x, pr['mean'], yerr=std, fmt = 'o', color=color)

    plot_distribution(pr.alpha, pr.beta, ax=ax, dist_type="beta", shaded="True", line_alpha=.3,
                    plot_kwargs={"color":color}, shade_alpha=.05,
                    vertical=True, fill_offset=(x), y_scale=.008)

    _ = hline_to_point(ax, x, pr['mean'], color=color, ls="--", alpha=.2)


if includem0:
    trials = ea.conditions['m0']
    for exclude, shortest, longest in zip(['left', 'right'], ['center', 'center'], ['right', 'left']):
        tleft = trials.loc[trials.escape_arm != exclude]
        pshort = len(tleft.loc[tleft.escape_arm==shortest])/len(tleft)
        lenratio = mazes['m0']["{}_path_length".format(longest)]/mazes['m0']["{}_path_length".format(shortest)]
        
        Xlabels.append("m0-{}\n{}".format(exclude, round(lenratio,1)))
        X.append(lenratio)
        color = maze_colors['m0']

        binary_trials = [1 if t.escape_arm == shortest else 0 for i,t in tleft.iterrows()]
        a2, b2, mean, mode, sigmasquared, prange = ea.grouped_bayes_analytical([len(binary_trials)], [np.sum(binary_trials)])
        ax.errorbar(lenratio, mean, yerr=math.sqrt(sigmasquared), fmt = 'o', color=color)


if includem6:
    pr = pRs.loc[pRs.condition == "m6"]
    if not use_eucl:
        x = mazes['m6']['ratio']
    else:
        x = combined_dists['m6']
            
    Xlabels.append("m6\n{}".format(round(x, 1)))
    X.append(x)
    # xfit.append(x); yfit.append(pr['mean']); stdfit.append(math.sqrt(pr.sigmasquared))

    ax.errorbar(x, pr['mean'], yerr=math.sqrt(pr.sigmasquared), fmt = 'o', color=maze_colors['m6'])
    plot_distribution(pr.alpha.values[0], pr.beta.values[0], ax=ax, dist_type="beta", shaded="True", line_alpha=.3,
                    plot_kwargs={"color":maze_colors['m6']}, shade_alpha=.05,
                    vertical=True, fill_offset=(x), y_scale=.008)



_ = ax.axhline(0.5, ls="--", color=[.5, .5, .5])

if not use_eucl: 
    xlim=[.5, 2.8]
else:
    xlim=[min(combined_dists.values())-.3,  max(combined_dists.values())+.3]

if fit_curve:
    curve_params = plot_fitted_curve(centered_logistic, xfit, yfit, ax, xrange=xlim, scatter_kwargs=dict(alpha=0),
                    fit_kwargs = dict(sigma=stdfit),
                    line_kwargs=dict(color=[.3, .3, .3], alpha=.7, lw=3))


_ = ax.set(title="p(R) for each maze", xticks=X, xlabel="Increasing asymmetry", 
                    xticklabels=Xlabels, 
                    ylabel="p(best)",
                    ylim=[0, 1], xlim=xlim ) #xlim=[.5, 2.8]
save_plot("psychometric_fivemaze", f)



# %%

# ---------------------------------------------------------------------------- #
#        !                        UTILITY SPACE                                #
# ---------------------------------------------------------------------------- #
cm = MplColorHelper("bwr", 0, 1)
pRs = ea.bayes_by_condition_analytical()

f, axarr = create_figure(subplots=True, ncols=2)

for condition in ea.conditions.keys():
    if condition == "m0": continue 
    pr = pRs.loc[pRs.condition == condition]['mean'].values[0]
    lr_ratio = mazes[condition]['ratio']
    eucl_dist = euclidean_dists[condition]
    time_ratio = path_durations[condition].ratio.mean
    axarr[0].scatter(lr_ratio, eucl_dist, color=maze_colors[condition], s=300, edgecolor=black, zorder=99)

surface = np.zeros((250, 250))
for n, i in enumerate(np.linspace(0, 3, 250)): # iterate geo ratio
    for k, ii in enumerate(np.linspace(0, 3, 250)): # iterate eucl ratio
        p = centered_logistic(0.75*i+ii, *curve_params)
        surface[n, k] = p
        # axarr[0].scatter(i, ii, color=cm.get_rgb(p), s=300)
axarr[0].imshow(surface, cmap="bwr", extent=[0, 3, 0, 3], origin="lower", vmin=0, vmax=1)
_ = axarr[0].set(xlim=[0,3], ylim=[0, 3])


# plot slices
for i in range(0, 250, 25):
    axarr[1].plot(surface[:, i], color=black)
    axarr[1].plot(surface[i, :], color=red)

save_plot("costfunc", f)









# %%
# ---------------------------------------------------------------------------- #
#                               ! TIMED ANALYSIS (time)                        #
# ---------------------------------------------------------------------------- #
# TODO try different window params
# TODO compare light vs dark

# ? params
n_random_iters = None
min_trials_in_bin = 4

for windows_size in [600]:
    f, axarr = create_figure(subplots=True, ncols=2, nrows=2, sharex=False, figsize=(12, 12))
    for n, (condition, trials) in enumerate(ea.conditions.items()):
        if condition not in psychometric_mazes_and_dark: continue
        n = psychometric_mazes_and_dark.index(condition)
        ax = axarr[n]
        trial_times = {'left':[], 'right':[]}
        tmax = 0

        # Get time and arm of escape for each trial
        for i, trial in trials.iterrows():
            if trial.escape_arm == "center": continue

            stim_time = trial.stim_frame_session/trial.fps
            trial_times[trial.escape_arm].append(stim_time)
            if stim_time > tmax: tmax = stim_time

        # bin trials by time
        x, y, yerr = [], [], []
        for i in range(int(np.ceil(tmax/windows_size))):
            # Get trials in time window
            ranges = (windows_size*i, windows_size*(i+1))
            x.append(np.mean(ranges))

            tr_in_t = {arm:[a for a in t if a>ranges[0] and a <= ranges[1]] for arm,t in trial_times.items()}
            binary_trials = np.hstack([np.ones(len(tr_in_t['right'])), np.zeros(len(tr_in_t['left']))])

            n_trials_in_bin = len(binary_trials)

            if n_trials_in_bin<min_trials_in_bin:
                y.append(np.nan)
                yerr.append(np.nan)
            else:
                a2, b2, mean, mode, sigmasquared, prange = ea.grouped_bayes_analytical(len(binary_trials), np.sum(binary_trials))
                y.append(mean)
                yerr.append(math.sqrt(sigmasquared))

            # random sampling
            if n_random_iters is not None:
                if n_trials_in_bin > min_trials_in_bin:
                    random_prs = []
                    for n in range(n_random_iters):
                        random_trials = trials.sample(n=n_trials_in_bin, axis=0)
                        random_trials = random_trials.loc[random_trials.escape_arm != "center"]
                        r_binary_trials = [1 if e == "right" else 0 for e in random_trials.escape_arm.values]
                        _, _, mean, _, sigmasquared, _ = ea.grouped_bayes_analytical([len(r_binary_trials)], [np.sum(r_binary_trials)])
                        random_prs.append(mean)
                    ax.errorbar(x[-1], np.mean(random_prs), yerr=np.std(random_prs), fmt="-", color=[.2, .2, .2], alpha=.85)
                    ax.scatter(x[-1], np.mean(random_prs), edgecolor=white, color=[.1, .1, .1], zorder=99)

        ax.errorbar(x, y, yerr=yerr, fmt="-", color=maze_colors[condition], alpha=.85)
        ax.scatter(x, y, edgecolor=black, color=maze_colors[condition], zorder=99)

        # mark the global average
        mn  = pRs.loc[pRs.condition==condition]['mean'].values[0]
        std = math.sqrt(pRs.loc[pRs.condition==condition]['sigmasquared'].values[0])
        rect = mpl.patches.Rectangle((0, mn-std), 10000, 2*std, lw=1, edgecolor=desaturate_color(maze_colors[condition]),
                    facecolor=desaturate_color(maze_colors[condition]), alpha=.1)
        ax.add_patch(rect)
        ax.axhline(mn, ls="--", color=desaturate_color(maze_colors[condition]), lw=4, alpha=.3)


        ax.axhline(.5, color=[.7, .7, .7], ls=":")
        ax.set(title="{} - {}s window".format(condition, windows_size), xlabel="time (min)", ylabel="p(R)", xlim=[0, 60*60*1.5],
                ylim=[0, 1], xticks=np.arange(0, 10000, 600), xticklabels=[int(x/60) for x in np.arange(0, 10000, 600)])

save_plot("timed_by_time", f)


# %%
# ---------------------------------------------------------------------------- #
#                          ! TIMED ANALYSIS (n trials)                         #
# ---------------------------------------------------------------------------- #
# TODO try different binning params
# TODO compare light vs dark
use_real_mean = False


for n_trials_in_bin in [20]:
    f, axarr = create_figure(subplots=True, ncols=2, nrows=2, sharex=False, figsize=(12, 12))
    for n, (condition, trials) in enumerate(ea.conditions.items()):
        if condition not in psychometric_mazes_and_dark: continue
        n = psychometric_mazes_and_dark.index(condition)

        ax = axarr[n]
        trial_times = []
        trial_outcomes = []
        tmax = 0

        # Get time and arm of escape for each trial
        for i, trial in trials.iterrows():
            if trial.escape_arm == "center": continue

            stim_time = trial.stim_frame_session/trial.fps
            trial_times.append(stim_time)
            if stim_time > tmax: tmax = stim_time
            if trial.escape_arm == "right": trial_outcomes.append(1)
            else: trial_outcomes.append(0)

        sort_idx = np.argsort(trial_times)
        trial_times = np.array(trial_times)[sort_idx]
        trial_outcomes = np.array(trial_outcomes)[sort_idx]
        n_trials = len(trial_outcomes)

        # # bin trials by n trials
        x, y, yerr = [], [], []
        for i in range(int(np.ceil(n_trials/n_trials_in_bin))):
            # Get trials in time window
            binary_trials = trial_outcomes[i*n_trials_in_bin:(i+1)*n_trials_in_bin]
            if len(binary_trials) < n_trials_in_bin: continue

            if not use_real_mean:
                a2, b2, mean, mode, sigmasquared, prange = ea.grouped_bayes_analytical(len(binary_trials), np.sum(binary_trials))
                y.append(mean)
                yerr.append(math.sqrt(sigmasquared))
            else:
                y.append(np.mean(binary_trials))
                yerr.append(np.std(binary_trials))

            x.append(i)

        ax.errorbar(x, y, yerr=yerr, fmt="-", color=maze_colors[condition], alpha=.85)
        ax.scatter(x, y, edgecolor=black, color=maze_colors[condition], zorder=99)

        # mark the global average
        mn  = pRs.loc[pRs.condition==condition]['mean'].values[0]
        std = math.sqrt(pRs.loc[pRs.condition==condition]['sigmasquared'].values[0])
        rect = mpl.patches.Rectangle((-1, mn-std), 10000, 2*std, lw=1, edgecolor=desaturate_color(maze_colors[condition]),
                    facecolor=desaturate_color(maze_colors[condition]), alpha=.1)
        ax.add_patch(rect)
        ax.axhline(mn, ls="--", color=desaturate_color(maze_colors[condition]), lw=4, alpha=.3)


        ax.axhline(.5, color=[.7, .7, .7], ls=":")
        ax.set(title="{} - {} trials bins".format(condition, n_trials_in_bin), xlabel="bin number", ylabel="p(R)", xlim=[-1, int(np.ceil(n_trials/n_trials_in_bin))],
                ylim=[0, 1], xticks=np.arange(0, 30, 5), xticklabels=np.arange(0, 30, 5))



    save_plot("timed_by_ntrials", f)





# %%
# ---------------------------------------------------------------------------- #
#                                     ! M6                                     #
# ---------------------------------------------------------------------------- #
# TODO test against NULL
pRs = ea.bayes_by_condition_analytical()


f, ax = create_figure(subplots=False)
pr = pRs.loc[pRs.condition=='m6']
n = len(ea.conditions['m6'])
k = np.int(n/2)

a2, b2, mean, mode, sigmasquared, prange = ea.grouped_bayes_analytical(n, k)

ax.errorbar(pr['mean'], 0*0.5*i, xerr=2*math.sqrt(pr.sigmasquared), color=maze_colors['m6'])
ax.scatter(pr['mean'], 0*0.5*i, color=maze_colors['m6'], s=250, edgecolor=black, zorder=99, label='m6')
plot_distribution(pr.alpha.values[0], pr.beta.values[0], ax=ax, dist_type="beta", shaded="True", line_alpha=.3,
                plot_kwargs={"color":maze_colors['m6']}, shade_alpha=.05, 
                y_scale=1)

ax.errorbar(mean, -.2*0.5*i, xerr=2*math.sqrt(sigmasquared), color=[.4, .4, .4])
ax.scatter(mean, -.2*0.5*i, color=[.4, .4, .4], s=250, edgecolor=black, zorder=99, label='null')
plot_distribution(a2, b2, ax=ax, dist_type="beta", shaded="True", line_alpha=.3,
                plot_kwargs={"color":[.4, .4, .4]}, shade_alpha=.05, 
                y_scale=1)

ax.legend()
_ = ax.axvline(.5, color=black, lw=2, ls=":")
_ = ax.set(title="M4 vs M6", xlabel="p(R)", ylabel="density", xlim=[0, 1])
save_plot("m6", f)









# %%

# %%
# ---------------------------------------------------------------------------- #
#                       OTHER THINGS NOT USED FREQUENTLY                       #
# ---------------------------------------------------------------------------- #

# ---------------------------- ! EFFECT OF ORIGIN ---------------------------- #
f, axarr = create_figure(subplots=True, ncols=2)

for i, (condition, trials) in enumerate(ea.conditions.items()):
    if condition in ["m6", "m0"]: continue

    sub_conds = dict(lori = trials.loc[trials.origin_arm=="left"],
                    rori = trials.loc[trials.origin_arm=="right"])

    prs = ea.bayes_by_condition_analytical(sub_conds)
    axarr[0].errorbar([0, 1 ], prs['mean'].values, yerr=np.sqrt(prs.sigmasquared.values),
                fmt = '-o', color=maze_colors[condition], label=condition)


    # Difference betweel lori and rori beta distributions
    ldist = get_distribution('beta', prs.loc[prs.condition=='lori'].alpha, prs.loc[prs.condition=='lori'].beta)
    rdist = get_distribution('beta', prs.loc[prs.condition=='rori'].alpha, prs.loc[prs.condition=='rori'].beta)
    delta = [l-r for l,r in zip(random.choices(ldist, k=50000), random.choices(rdist, k=50000))]
    percdelta = percentile_range(delta)

    axarr[1].hist(delta, bins=30, color=maze_colors[condition], edgecolor=maze_colors[condition],  
                alpha=.1, histtype="stepfilled", density=True)
    axarr[1].hist(delta, bins=30, color=maze_colors[condition], edgecolor=maze_colors[condition],  
                label=condition, alpha=1, histtype="step", linewidth=3, density=True)

    axarr[1].errorbar(np.mean(delta), -.5-.5*i, xerr=percdelta.mean-percdelta.low, lw=4, fmt="o",
                color=maze_colors[condition])
    axarr[1].scatter(np.mean(delta), -.5-.5*i, s=100, edgecolor=black, color=maze_colors[condition], zorder=99)


axarr[0].axhline(.5, lw=1, color=black, alpha=.4, ls=":")
ortholines(axarr[0], [1, 1,], [0, 1], lw=2, color=black, alpha=.1, ls="--")
axarr[0].legend()

axarr[1].axvline(0, lw=2, color=black, alpha=.4, ls="--")


_ = axarr[0].set(title="p(R) vs arm of origin", xlabel="origin", xticks=[0, 1], 
        xticklabels=["left", "right"], ylim=[.3, 1])
_ = axarr[1].set(title="p(R|L) - p(R|R)", xticks=[-.2, 0, .2], 
        xticklabels=["-0.2\nR > L", "same", "0.2\nL > R"], ylabel="density")
save_plot("effect_origin", f)


# %%
# -------------------------- ! EXPLORATION ANALYSIS -------------------------- #
# exploration_data, alldata = ea.get_exploration_per_path_from_trials()
xticks, xlabels = [], []
f, ax = create_figure(subplots=False)
for i, (condition, data) in enumerate(exploration_data.items()):
    if condition == "m0": continue
    x = [i-.25, i+.25]
    xticks.extend([i-.25,i, i+.25])
    xlabels.extend(["left","\n{}".format(condition), "right"])
    y = [data.left.median/mazes[condition]['ratio'], data.right.median]
    yerr = [data.left.sem, data.right.sem]

    ax.axvline(i-.25, color=[.2, .2, .2], ls=":", alpha=.15)
    ax.axvline(i+.25, color=[.2, .2, .2], ls=":", alpha=.15)

    ax.plot(x, y, "-o", label=condition, color=maze_colors[condition], zorder=90)
    ax.scatter(x, y, edgecolor=black, s=250, color=maze_colors[condition], zorder=99)
    ax.errorbar(x, y, yerr, color=maze_colors[condition], zorder=90)


    ttest, pval = stats.ttest_ind([x/mazes[condition]['ratio'] for x in  alldata[condition]['left'][0]], 
                                    alldata[condition]['right'][0])
    if pval < .05:
        ax.plot([i-.3, i+.3], [350, 350], lw=4, color=[.4, .4, .4])
        ax.text(i-0.025, 350, "*", fontsize=20)
    else:
        ax.plot([i-.3, i+.3], [350, 350], lw=4, color=[.7, .7, .7])
        ax.text(i-0.05, 353, "n.s.", fontsize=16)

_ = ax.set(title="Normalized arm occupancy during exploration", xticks=xticks, ylim=[0, 380],
        xticklabels=xlabels, ylabel="norm. occupancy (s/len)")
save_plot("effect_exploration", f)



# %%
# ----------------------------- ! LIGHT VS DARK ----------------------------- #
# Light vs dark -- get data
# ea2 = ExperimentsAnalyser(load_psychometric=False, tracking="all")
# ea2.max_duration_th = 9

# ea2.add_condition("m1-light", maze_design=1, lights=1, escapes_dur=True, tracking="all")
# ea2.add_condition("m1-dark", maze_design=1, lights=0, escapes_dur=True, tracking="all")

# for condition, trials in ea2.conditions.items():
#     print("Maze {} -- {} trials".format(condition, len(trials)))


# Calc and plot pR for light vs dark
pRs = ea2.bayes_by_condition_analytical()

f, axarr = create_figure(subplots=True, ncols=2)

X = [1, 1.15]
cols = [maze_colors["m1"], desaturate_color(maze_colors["m1"], k=.2)]
for i, pr in pRs.iterrows():
    std = math.sqrt(pr.sigmasquared)

    axarr[0].errorbar(pr['median'], -.5, xerr=std, fmt = 'o', color=cols[i])
    axarr[0].scatter(pr['median'], -.5, edgecolor=black, color=cols[i])

    plot_distribution(pr.alpha, pr.beta, ax=axarr[0], dist_type="beta", shaded="True", line_alpha=.8,
                    plot_kwargs={"color":cols[i]}, shade_alpha=.1, y_scale=1)

    # _ = vline_to_point(axarr[0], X[i], pr['median'], color=cols[i], ls="--", alpha=.2)
# _ = axarr[0].axvline(0.5, ls="--", color=[.5, .5, .5])


# Difference betweel lori and rori beta distributions
# ldist = get_distribution('beta', pRs.loc[pRs.condition=='m1-light'].alpha, pRs.loc[pRs.condition=='m1-light'].beta)
# ddist = get_distribution('beta', pRs.loc[pRs.condition=='m1-dark'].alpha, pRs.loc[pRs.condition=='m1-dark'].beta)
# delta = [d-l for l,d in zip(random.choices(ldist, k=50000), random.choices(ddist, k=50000))]
# percdelta = percentile_range(delta)

# axarr[1].hist(delta, bins=30, color=cols[0], edgecolor=cols[0],  
#             alpha=.1, histtype="stepfilled", density=True)
# axarr[1].hist(delta, bins=30, color=cols[0], edgecolor=cols[0],  
#             alpha=1, histtype="step", linewidth=3, density=True)

# axarr[1].errorbar(np.mean(delta), -1.5, xerr=percdelta.mean-percdelta.low, lw=4, fmt="o",
#             color=cols[0])
# axarr[1].scatter(np.mean(delta), -1.5, s=100, edgecolor=black, color=cols[0], zorder=99)
# _ = axarr[1].axvline(0, ls="--", color=[.5, .5, .5])


_ = axarr[0].set(title="p(R) dark vs light", ylabel='density', 
                    xlabel="p(R)",xlim=[0.5, 1])
_ = axarr[1].set(title="p(R|dark) - p(R|light)", xticks=[0, 0.1, 0.2], xticklabels=[0, 0.1, 0.2], 
                    ylabel="pdensity",)

save_plot("effect_light", f)

# %%
# -------------------------- ! NAVIE VS EXPERIENCED -------------------------- #
# Light vs dark -- get data
ea3 = ExperimentsAnalyser(load_psychometric=False, tracking="all")
ea3.max_duration_th = 9

ea3.add_condition("m1-naive", maze_design=1, naive=1, lights=None, escapes_dur=True, tracking="all")
ea3.add_condition("m1-exper", maze_design=1, naive=0, lights=None, escapes_dur=True, tracking="all")

for condition, trials in ea3.conditions.items():
    print("Maze {} -- {} trials".format(condition, len(trials)))

# Calc and plot pR for light vs dark
pRs = ea3.bayes_by_condition_analytical()

f, axarr = create_figure(subplots=True, ncols=2)

X = [1, 1.15]
cols = [maze_colors["m1"], desaturate_color(maze_colors["m1"])]
for i, pr in pRs.iterrows():
    std = math.sqrt(pr.sigmasquared)

    axarr[0].errorbar(X[i], pr['median'], yerr=std,fmt = 'o', color=cols[i])

    plot_distribution(pr.alpha, pr.beta, ax=axarr[0], dist_type="beta", shaded="True", line_alpha=.2,
                    plot_kwargs={"color":cols[i]}, shade_alpha=.05,
                    vertical=True, fill_offset=(X[i]), y_scale=.012)

    _ = hline_to_point(axarr[0], X[i], pr['median'], color=cols[i], ls="--", alpha=.2)
_ = axarr[0].axhline(0.5, ls="--", color=[.5, .5, .5])


# Difference betweel lori and rori beta distributions
ldist = get_distribution('beta', pRs.loc[pRs.condition=='m1-naive'].alpha, pRs.loc[pRs.condition=='m1-naive'].beta)
ddist = get_distribution('beta', pRs.loc[pRs.condition=='m1-exper'].alpha, pRs.loc[pRs.condition=='m1-exper'].beta)
delta = [d-l for l,d in zip(random.choices(ldist, k=50000), random.choices(ddist, k=50000))]
percdelta = percentile_range(delta)

axarr[1].hist(delta, bins=30, color=cols[0], edgecolor=cols[0],  
            alpha=.1, histtype="stepfilled", density=True)
axarr[1].hist(delta, bins=30, color=cols[0], edgecolor=cols[0],  
            alpha=1, histtype="step", linewidth=3, density=True)

axarr[1].errorbar(np.mean(delta), -1.5, xerr=percdelta.mean-percdelta.low, lw=4, fmt="o",
            color=cols[0])
axarr[1].scatter(np.mean(delta), -1.5, s=100, edgecolor=black, color=cols[0], zorder=99)
_ = axarr[1].axvline(0, ls="--", color=[.5, .5, .5])


_ = axarr[0].set(title="p(R) naive vs exper.", xticks=X, xticklabels=['naive', 'exper.'], 
                    ylabel="p(R)", ylim=[.3, 1], xlim=[0.95, 1.35])

save_plot("effect_naive", f)


# %%
