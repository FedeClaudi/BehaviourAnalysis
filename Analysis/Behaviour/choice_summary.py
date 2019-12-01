# %%
import sys
sys.path.append('./')
from Utilities.imports import *
import statsmodels.api as sm
from pandas.plotting import scatter_matrix

from scipy.optimize import curve_fit


from Analysis.Behaviour.utils.experiments_analyser import ExperimentsAnalyser
from Processing.rois_toolbox.rois_stats import convert_roi_id_to_tag


# %%
# Define a bunch of useful colors
%matplotlib inline
palette = makePalette(green, orange, 5, False)
maze_colors = {
    'm1': palette[0],
    'm2': palette[1],
    'm3': palette[2],
    'm4': palette[3],
    'm6':lilla
}

arms_colors = {
    "left": lilla,
    "right": teal,
}

psychometric_mazes = ["m1", "m2", "m3", "m4"]

# %%
# Getting data
ea = ExperimentsAnalyser(load_psychometric=False, tracking="all")
ea.max_duration_th = 9

ea.add_condition("m1", maze_design=1, lights=None, escapes_dur=True, tracking="all")
ea.add_condition("m2", maze_design=2, lights=None, escapes_dur=True, tracking="all")
ea.add_condition("m3", maze_design=3, lights=None, escapes_dur=True, tracking="all")
ea.add_condition("m4", maze_design=4, lights=None, escapes_dur=True, tracking="all")
ea.add_condition("m6", maze_design=6, lights=None, escapes_dur=True, tracking="all")

for condition, trials in ea.conditions.items():
    print("Maze {} -- {} trials".format(condition, len(trials)))


# %%
# inspect stuff
f, axarr = create_figure(subplots=True, ncols=3)
for condition, trials in ea.conditions.items():
    mean_speed = [np.nanmean(t.body_speed) for i,t in trials.iterrows()]
    axarr[2].scatter(trials.escape_duration.values, mean_speed, color=maze_colors[condition], alpha=.4, label=condition)

    axarr[0].hist(mean_speed, histtype="stepfilled", alpha=.25,
                         color=maze_colors[condition], density=True)

    axarr[1].hist(trials.escape_duration.values, histtype="stepfilled", alpha=.25,
                         color=maze_colors[condition], density=True)

    print("{} - mean speed {}+-{} -- {}".format(condition, round(np.mean(mean_speed), 2), 
                                round(np.std(mean_speed), 3), round(np.mean(mean_speed)+2*np.std(mean_speed), 3)))
    axarr[0].axvline(np.mean(mean_speed), color=maze_colors[condition])

axarr[1].axvline(12, color=black, ls=":")
axarr[2].axvline(12, color=black, ls=":")

axarr[2].legend()
_ = axarr[2].set(title="dur vs speed", xlabel="escape duration (s)", ylabel="mean escape speed")
axarr[0].set(title="mean speed")
axarr[1].set(title="escape duration")




# %%
# ! PATH LENGTH
# Calc and plot path lengths for each condition
path_lengths = ea.get_arms_lengths_from_trials()

f, axarr = create_figure(subplots=True, ncols=2)
for i, (condition, lengths) in enumerate(path_lengths.items()):
    # if condition == "m6": continue
    x = [0, 1]
    y = [lengths.left.mean, lengths.right.mean]
    yerr = np.array([[lengths.left.std, lengths.left.std], [lengths.right.std, lengths.right.std]]).reshape(2, 2)
    axarr[0].errorbar(x, y, yerr=yerr, fmt="-o", label=condition, color=maze_colors[condition])
    axarr[1].scatter(1-0.1*i, lengths.ratio.mean, s=50, 
                    label=condition, color=maze_colors[condition])
    _ = vline_to_point(axarr[1], 1-0.1*i, lengths.ratio.mean,color=maze_colors[condition], ymin=0, ls="--", alpha=.5)
    _ = hline_to_point(axarr[1], 1-0.1*i, lengths.ratio.mean,color=maze_colors[condition], ls="--", alpha=.5)

_ = axarr[1].axhline(1, ls="--", color=[.5, .5, .5])
_ = axarr[0].legend()
_ = axarr[0].set(title="Paths lengths per maze design", xticks=[0, 1], xticklabels=["left", "right"],
                xlim=[-.1, 1.1], ylabel="mean length (a.u.)")
_ = axarr[1].set(title="Path lengths ratio", xticks=[1-0.1*i for i in range(len(psychometric_mazes))],
                xticklabels=psychometric_mazes, ylim=[.75, 2], ylabel="ratio L/R", xlim=[0.5, 1.1])


# %%
# ! ESCAPE DURATION
path_durations = ea.get_duration_per_arm_from_trials()

f, axarr = create_figure(subplots=True, ncols=2)
for i, (condition, durations) in enumerate(path_durations.items()):
    # if condition == "m6": continue
    x = [0, 1]
    y = [durations.left.mean, durations.right.mean]
    axarr[0].plot(x, y, "-o", label=condition, color=maze_colors[condition])
    axarr[1].scatter(1-0.1*i, durations.ratio.mean, s=50, 
                    label=condition, color=maze_colors[condition])
    _ = vline_to_point(axarr[1], 1-0.1*i, durations.ratio.mean,color=maze_colors[condition], ymin=0, ls="--", alpha=.5)
    _ = hline_to_point(axarr[1], 1-0.1*i, durations.ratio.mean,color=maze_colors[condition], ls="--", alpha=.5)

_ = axarr[1].axhline(1, ls="--", color=[.5, .5, .5])
_ = axarr[0].legend()
_ = axarr[0].set(title="Paths durations per maze design", xticks=[0, 1], xticklabels=["left", "right"],
                xlim=[-.1, 1.1], ylabel="mean duration (s)")
_ = axarr[1].set(title="Path durations ratio", xticks=[1-0.1*i for i in range(len(psychometric_mazes))],
                xticklabels=psychometric_mazes, ylim=[.75, 2.5], ylabel="ratio L/R", xlim=[0.5, 1.1])


# %%
# ! DURATION VS LENGTH
f, ax = create_figure(subplots=False)

all_l, all_r = [[], []], [[], []]
for condition, trials in ea.conditions.items():
    if condition == "m6": continue
    dd, ll, cc = [], [], []
    for i, trial in trials.iterrows():
        d = trial.after_t_tracking.shape[0]/trial.fps
        l = np.sum(calc_distance_between_points_in_a_vector_2d(trial.after_t_tracking))
        dd.append(d)
        ll.append(l)

        if trial.escape_arm == "right" or condition=='m4': 
            cc.append(arms_colors['right'])
            all_r[0].append(l)
            all_r[1].append(d)
        else:
            cc.append(arms_colors['left'])
            all_l[0].append(l)
            all_l[1].append(d)
            # ax.scatter(l, d, color=black, s=40, zorder=99)


    ax.scatter(ll, dd, color=cc, alpha=.3, s=50)

# _ = ax.legend()
_ = ax.set(title="Avg path length vs Avg duration", xlabel="length (a.u.)", ylabel="duration (s)")



# %%
# ! PSYCHOMETRIC

# Calc and plot pR for psychometric data
pRs = ea.bayes_by_condition_analytical()

f, ax = create_figure(subplots=False)

X, Y = [], []
for i, pr in pRs.iterrows():
    if pr.condition == "m6": continue
    std = math.sqrt(pr.sigmasquared)
    yerr = np.array([pr['mean']-pr.prange.low, pr.prange.high-pr['mean']]).reshape(2, 1)
    x = path_lengths[pr.condition].ratio.mean
    X.append(x)
    Y.append(pr['mean'])
    color=maze_colors[pr.condition]

    ax.errorbar(x, pr['mean'], yerr=std, fmt = 'o', color=color)

    plot_distribution(pr.alpha, pr.beta, ax=ax, dist_type="beta", shaded="True", line_alpha=.3,
                    plot_kwargs={"color":color}, shade_alpha=.05,
                    vertical=True, fill_offset=(x), y_scale=.008)

    _ = hline_to_point(ax, x, pr['mean'], color=color, ls="--", alpha=.2)

plot_fitted_curve(logistic, X, Y, ax, xrange=[0, 3], scatter_kwargs=dict(alpha=0),
                line_kwargs=dict(color=[.3, .3, .3], alpha=.7, lw=3))
_ = ax.axhline(0.5, ls="--", color=[.5, .5, .5])

_ = ax.set(title="p(R) for each maze", xticks=X, xlabel="L/R length ratio", xticklabels=psychometric_mazes, 
                    ylabel="p(R)",
                    ylim=[0, 1], xlim=[.5, 2.2])


# %%
# ! TIMED ANALYSIS
# ? params
n_random_iters = 50
for windows_size in [300]:
    # windows_size = 600 # window size in seconds
    min_trials_in_bin = 1

    f, axarr = create_figure(subplots=True, ncols=2, nrows=2, sharex=False, figsize=(12, 12))
    for n, (condition, trials) in enumerate(ea.conditions.items()):
        if condition == "m6": continue
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
                a2, b2, mean, mode, sigmasquared, prange = ea.grouped_bayes_analytical([len(binary_trials)], [np.sum(binary_trials)])
                y.append(mean)
                yerr.append(math.sqrt(sigmasquared))

            # random sampling
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
        ax.set(title="{} - {}s window".format(condition, windows_size), xlabel="time (s)", ylabel="p(R)", xlim=[0, 60*60*1.5],
                ylim=[0, 1], xticks=np.arange(0, 10000, 600), xticklabels=[int(x/60) for x in np.arange(0, 10000, 600)])

# %%
# ! timed analysis test
n_random_iters = 50
windows_size = 300 
n_trials_in_bin = 20

f, axarr = create_figure(subplots=True, ncols=2, nrows=2, sharex=False, figsize=(12, 12))
for n, (condition, trials) in enumerate(ea.conditions.items()):
    if condition == "m6": continue
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

        # a2, b2, mean, mode, sigmasquared, prange = ea.grouped_bayes_analytical([len(binary_trials)], [np.sum(binary_trials)])
        y.append(np.mean(binary_trials))
        x.append(i)
        yerr.append(np.std(binary_trials))

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



    
# %%
# ! EFFECT OF ORIGIN
f, axarr = create_figure(subplots=True, ncols=2)

for i, (condition, trials) in enumerate(ea.conditions.items()):
    if condition == "m6": continue

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

# %%
# ! EXPLORATION ANALYSIS
# exploration_data = ea.get_exploration_per_path_from_trials()
f, ax = create_figure(subplots=False)
for condition, data in exploration_data.items():
    if condition == "m6": continue
    x = [0, 1]
    y = [data.left.median/path_lengths[condition].ratio.mean, 
        data.right.median]

    ax.plot(x, y, color=maze_colors[condition])
    ax.scatter(x, y, s=250, edgecolor=black, color=maze_colors[condition], label=condition)

ax.legend()
_ = ax.set(title="Normalized arm occupancy during exploration", xlabel="arm", xticks=[0, 1], 
        xticklabels=["left", "right"], ylabel="norm. occupancy (s/len)")



# %%
# ! M6

pRs = ea.bayes_by_condition_analytical()

conditions = ['m4', 'm6']
f, ax = create_figure(subplots=False)
for i, condition in enumerate(conditions):
    pr = pRs.loc[pRs.condition==condition]

    ax.errorbar(pr['mean'], 0*0.5*i, xerr=math.sqrt(pr.sigmasquared), color=maze_colors[condition])
    ax.scatter(pr['mean'], 0*0.5*i, color=maze_colors[condition], s=250, edgecolor=black, zorder=99, label=condition)


    plot_distribution(pr.alpha.values[0], pr.beta.values[0], ax=ax, dist_type="beta", shaded="True", line_alpha=.3,
                    plot_kwargs={"color":maze_colors[condition]}, shade_alpha=.05, 
                    y_scale=.01)

ax.legend()
_ = ax.axvline(.5, color=black, lw=2, ls=":")
_ = ax.set(title="M4 vs M6", xlabel="p(R)", ylabel="density", xlim=[0, 1])
    
# %%
# ! M6 TRACKING 

f, ax = create_figure(subplots=False)
for i, trial in ea.conditions['m3'].iterrows():
    ax.plot(trial.body_xy[:, 0], trial.body_xy[:, 1])




# %%
# ! LIGHT VS DARK
# Light vs dark -- get data
ea2 = ExperimentsAnalyser(load_psychometric=False, tracking="all")
ea2.max_duration_th = 9

ea2.add_condition("m1-light", maze_design=1, lights=1, escapes_dur=True, tracking="all")
ea2.add_condition("m1-dark", maze_design=1, lights=0, escapes_dur=True, tracking="all")

for condition, trials in ea2.conditions.items():
    print("Maze {} -- {} trials".format(condition, len(trials)))
# %%
# Calc and plot pR for light vs dark
pRs = ea2.bayes_by_condition_analytical()

f, axarr = create_figure(subplots=True, ncols=2)

X = [1, 1.15]
cols = [maze_colors["m1"], desaturate_color(maze_colors["m1"])]
for i, pr in pRs.iterrows():
    std = math.sqrt(pr.sigmasquared)

    axarr[0].errorbar(X[i], pr['median'], yerr=std, fmt = 'o', color=cols[i])

    plot_distribution(pr.alpha, pr.beta, ax=axarr[0], dist_type="beta", shaded="True", line_alpha=.2,
                    plot_kwargs={"color":cols[i]}, shade_alpha=.05,
                    vertical=True, fill_offset=(X[i]), y_scale=.012)

    _ = hline_to_point(axarr[0], X[i], pr['median'], color=cols[i], ls="--", alpha=.2)
_ = axarr[0].axhline(0.5, ls="--", color=[.5, .5, .5])


# Difference betweel lori and rori beta distributions
ldist = get_distribution('beta', pRs.loc[pRs.condition=='m1-light'].alpha, pRs.loc[pRs.condition=='m1-light'].beta)
ddist = get_distribution('beta', pRs.loc[pRs.condition=='m1-dark'].alpha, pRs.loc[pRs.condition=='m1-dark'].beta)
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


_ = axarr[0].set(title="p(R) dark vs light", xticks=X, xticklabels=['light', 'dark'], 
                    ylabel="p(R)", ylim=[.3, 1], xlim=[0.95, 1.35])
_ = axarr[1].set(title="p(R|dark) - p(R|light)", xticks=[0, 0.1, 0.2], xticklabels=[0, 0.1, 0.2], 
                    ylabel="pdensity",)


# TODO CHECK IF IN DARK TRIALS MICE EXPLORE MORE

# %%
# ! NAVIE VS EXPERIENCED
# Light vs dark -- get data
ea3 = ExperimentsAnalyser(load_psychometric=False, tracking="all")
ea3.max_duration_th = 9

ea3.add_condition("m1-naive", maze_design=1, naive=1, lights=None, escapes_dur=True, tracking="all")
ea3.add_condition("m1-exper", maze_design=1, naive=0, lights=None, escapes_dur=True, tracking="all")

for condition, trials in ea3.conditions.items():
    print("Maze {} -- {} trials".format(condition, len(trials)))
# %%
# Calc and plot pR for light vs dark
pRs = ea3.bayes_by_condition_analytical()

f, ax = create_figure(subplots=False)

X = [1, 1.15]
cols = [maze_colors["m1"], desaturate_color(maze_colors["m1"])]
for i, pr in pRs.iterrows():
    std = math.sqrt(pr.sigmasquared)

    ax.errorbar(X[i], pr['median'], yerr=std,fmt = 'o', color=cols[i])

    plot_distribution(pr.alpha, pr.beta, ax=ax, dist_type="beta", shaded="True", line_alpha=.2,
                    plot_kwargs={"color":cols[i]}, shade_alpha=.05,
                    vertical=True, fill_offset=(X[i]), y_scale=.012)

    _ = hline_to_point(ax, X[i], pr['median'], color=cols[i], ls="--", alpha=.2)
_ = ax.axhline(0.5, ls="--", color=[.5, .5, .5])

_ = ax.set(title="p(R) naive vs exper.", xticks=X, xticklabels=['naive', 'exper.'], 
                    ylabel="p(R)", ylim=[.3, 1], xlim=[0.95, 1.35])

