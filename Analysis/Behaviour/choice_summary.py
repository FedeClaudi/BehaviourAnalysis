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
# Getting data
ea = ExperimentsAnalyser(load_psychometric=False, tracking="all")
ea.max_duration_th = 9

ea.add_condition("m1", maze_design=1, lights=1, escapes_dur=True, tracking="all")
ea.add_condition("m2", maze_design=2, lights=1, escapes_dur=True, tracking="all")
ea.add_condition("m3", maze_design=3, lights=1, escapes_dur=True, tracking="all")
ea.add_condition("m4", maze_design=4, lights=1, escapes_dur=True, tracking="all")
ea.add_condition("m6", maze_design=6, lights=1, escapes_dur=True, tracking="all")

for condition, trials in ea.conditions.items():
    print("MAze {} -- {} trials".format(condition, len(trials)))


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
                    ylim=[.3, 1], xlim=[.85, 2.1])





# %%
