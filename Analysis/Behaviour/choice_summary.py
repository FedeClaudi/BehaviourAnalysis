# %%
import sys
sys.path.append('./')
from Utilities.imports import *
import statsmodels.api as sm
from pandas.plotting import scatter_matrix

from Analysis.Behaviour.utils.experiments_analyser import ExperimentsAnalyser

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

psychometric_mazes = ["m1", "m2", "m3", "m4"]

# %%
# ! PSYCHOMETRIC
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
# Calc and plot path lengths for each condition
path_lengths = ea.get_arms_lengths_from_trials()

f, axarr = create_figure(subplots=True, ncols=2)
for i, (condition, lengths) in enumerate(path_lengths.items()):
    if condition == "m6": continue
    x = [0, 1]
    y = [lengths.left.mean, lengths.right.mean]
    axarr[0].plot(x, y, "-o", label=condition, color=maze_colors[condition])
    axarr[1].scatter(1-0.1*i, lengths.ratio.mean, s=50, 
                    label=condition, color=maze_colors[condition])
    _ = vline_to_point(axarr[1], 1-0.1*i, lengths.ratio.mean,color=maze_colors[condition], ymin=0, ls="--", alpha=.5)
    _ = hline_to_point(axarr[1], 1-0.1*i, lengths.ratio.mean,color=maze_colors[condition], ls="--", alpha=.5)

_ = axarr[1].axhline(1, ls="--", color=[.5, .5, .5])
_ = axarr[0].legend()
_ = axarr[0].set(title="Paths lengths per maze design", xticks=[0, 1], xticklabels=["left", "right"],
                xlim=[-.1, 1.1], ylabel="(a.u.)")
_ = axarr[1].set(title="Path lengths ratio", xticks=[1-0.1*i for i in range(len(psychometric_mazes))],
                xticklabels=psychometric_mazes, ylim=[.75, 2], ylabel="ratio", xlim=[0.5, 1.1])


# %%
# Calc and plot pR for psychometric data
pRs = ea.bayes_by_condition_analytical()

f, ax = create_figure(subplots=False)

# TODO fit sigmoid

X = []
for i, pr in pRs.iterrows():
    if pr.condition == "m6": continue
    std = math.sqrt(pr.sigmasquared)
    yerr = np.array([pr['median']-pr.prange.low, pr.prange.high-pr['median']]).reshape(2, 1)
    x = path_lengths[pr.condition].ratio.mean
    X.append(x)
    color=maze_colors[pr.condition]

    ax.errorbar(x, pr['median'], yerr=std,
                fmt = 'o', color=color)

    plot_distribution(pr.alpha, pr.beta, ax=ax, dist_type="beta", shaded="True", line_alpha=.2,
                    plot_kwargs={"color":color}, shade_alpha=.05,
                    vertical=True, fill_offset=(x), y_scale=.012)


    _ = hline_to_point(ax, x, pr['median'], color=color, ls="--", alpha=.2)
_ = ax.axhline(0.5, ls="--", color=[.5, .5, .5])

_ = ax.set(title="p(R) for each maze", xticks=X, xlabel="L/R length ratio", xticklabels=psychometric_mazes, 
                    ylabel="p(R)",
                    ylim=[.3, 1], xlim=[.85, 2.2])





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

_ = ax.set(title="p(R) dark vs light", xticks=X, xticklabels=['light', 'dark'], 
                    ylabel="p(R)", ylim=[.3, 1], xlim=[0.95, 1.35])



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


# %%
# ! EFFECT OF ORIGIN
f, axarr = create_figure(subplots=True, ncols=2)

for i, (condition, trials) in enumerate(ea.conditions.items()):
    if condition == "m6": continue

    sub_conds = dict(lori = trials.loc[trials.origin_arm=="left"],
                    all=trials,
                    rori = trials.loc[trials.origin_arm=="right"])

    prs = ea.bayes_by_condition_analytical(sub_conds)
    axarr[0].errorbar([0, 1 ,2], prs['mean'].values, yerr=np.sqrt(prs.sigmasquared.values),
                fmt = '-o', color=maze_colors[condition], label=condition)


    # Difference betweel lori and rori beta distributions
    ldist = get_distribution('beta', prs.loc[prs.condition=='lori'].alpha, prs.loc[prs.condition=='lori'].beta)
    rdist = get_distribution('beta', prs.loc[prs.condition=='rori'].alpha, prs.loc[prs.condition=='rori'].beta)
    delta = [l-r for l,r in zip(random.choices(ldist, k=50000), random.choices(rdist, k=50000))]

    axarr[1].hist(delta, bins=30, color=maze_colors[condition], edgecolor=desaturate_color(maze_colors[condition]),  
                alpha=.1, histtype="stepfilled", density=True)
    axarr[1].hist(delta, bins=30, color=maze_colors[condition], edgecolor=desaturate_color(maze_colors[condition]),  
                label=condition, alpha=1, histtype="step", linewidth=3, density=True)
    axarr[1].errorbar(np.mean(delta), -.5-.15*i, xerr=np.std(delta), lw=2, fmt="o",
                color=desaturate_color(maze_colors[condition])) # TODO change this to CI


axarr[0].axhline(.5, lw=1, color=black, alpha=.4, ls=":")
ortholines(axarr[0], [1, 1, 1], [0, 1, 2], lw=2, color=black, alpha=.1, ls="--")
axarr[0].legend()

axarr[1].axvline(0, lw=2, color=black, alpha=.4, ls="--")



_ = axarr[0].set(title="p(R) vs arm of origin", xlabel="origin", xticks=[0, 1, 2], 
        xticklabels=["left", "both", "right"], ylim=[.3, 1])
_ = axarr[1].set(title="p(R|L) - p(R|R)", xticks=[-.2, 0, .2], 
        xticklabels=["R > L", "same", "L > R"], ylabel="density")



# %%
