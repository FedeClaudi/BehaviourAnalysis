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
# ! PATH LENGTH
# Calc and plot path lengths for each condition
path_lengths = ea.get_arms_lengths_from_trials()

f, axarr = create_figure(subplots=True, ncols=2)
for i, (condition, lengths) in enumerate(path_lengths.items()):
    if condition == "m6": continue
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
    if condition == "m6": continue
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

# sns.regplot(all_l[0], all_l[1], scatter=False, 
#                     truncate=True, color=arms_colors['left'], robust=False, ax=ax)
# sns.regplot(all_r[0], all_r[1], scatter=False, 
#                     truncate=True, color=arms_colors['right'], robust=False, ax=ax)

# for (condition, lengths), (_, durations) in zip(path_lengths.items(), path_durations.items()):
#     if condition == "m6": continue
#     ax.scatter(lengths.left.median, durations.left.median, color=maze_colors[condition], s=150, label=condition+" (left)", 
#                 zorder=99, edgecolor=black)



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
    yerr = np.array([pr['median']-pr.prange.low, pr.prange.high-pr['median']]).reshape(2, 1)
    x = path_lengths[pr.condition].ratio.mean
    X.append(x)
    Y.append(pr['median'])
    color=maze_colors[pr.condition]

    ax.errorbar(x, pr['median'], yerr=std, fmt = 'o', color=color)

    plot_distribution(pr.alpha, pr.beta, ax=ax, dist_type="beta", shaded="True", line_alpha=.3,
                    plot_kwargs={"color":color}, shade_alpha=.05,
                    vertical=True, fill_offset=(x), y_scale=.008)

    _ = hline_to_point(ax, x, pr['median'], color=color, ls="--", alpha=.2)

plot_fitted_curve(logistic, X, Y, ax, xrange=[0, 3], scatter_kwargs=dict(alpha=0),
                line_kwargs=dict(color=[.3, .3, .3], alpha=.7, lw=3))
_ = ax.axhline(0.5, ls="--", color=[.5, .5, .5])

_ = ax.set(title="p(R) for each maze", xticks=X, xlabel="L/R length ratio", xticklabels=psychometric_mazes, 
                    ylabel="p(R)",
                    ylim=[0, 1], xlim=[.5, 2.2])



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

