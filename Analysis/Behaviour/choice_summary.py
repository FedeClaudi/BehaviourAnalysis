# %%
import sys
sys.path.append('./')
from Utilities.imports import *
import statsmodels.api as sm
from pandas.plotting import scatter_matrix

from scipy.optimize import curve_fit
from scipy import signal


from Analysis.Behaviour.utils.experiments_analyser import ExperimentsAnalyser
from Processing.rois_toolbox.rois_stats import convert_roi_id_to_tag

def save_plot(name, f):
    if sys.platform == 'darwin': 
        fld = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/plots/choice_summary"
    else:
        fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\plots\\choice_summary"
    f.savefig(os.path.join(fld,"svg", "{}.svg".format(name)))
    f.savefig(os.path.join(fld, "{}.png".format(name)))


# %%
# Define a bunch of useful colors
%matplotlib inline
palette = makePalette(green, orange, 5 , False)
maze_colors = {
    'm0': darkgreen,
    'm1': palette[0],
    'm2': palette[1],
    'm3': palette[2],
    'm4': palette[3],
    'm6': salmon,
}

palette = makePalette(teal, lilla, 4, False)
arms_colors = {
    "left": teal,
    'center': darkcyan,
    "right": lilla,
}



psychometric_mazes = ["m1", "m2", "m3", "m4"]
five_mazes = ["m1", "m2", "m3", "m4", "m6"]
m6 = ["m6"]
m0 = ["m0"]
arms = ['left', 'right', 'center']

# TODO write functino that gets N,n,M for each experiment
# %%
# Getting data
ea = ExperimentsAnalyser(load_psychometric=False, tracking="all")
ea.max_duration_th = 12

ea.add_condition("m0", maze_design=0, lights=None, escapes_dur=True, tracking="all"); print("Got m0")
ea.add_condition("m1", maze_design=1, lights=None, escapes_dur=True, tracking="all"); print("Got m1")
ea.add_condition("m2", maze_design=2, lights=None, escapes_dur=True, tracking="all"); print("Got m2")
ea.add_condition("m3", maze_design=3, lights=None, escapes_dur=True, tracking="all"); print("Got m3")
ea.add_condition("m4", maze_design=4, lights=None, escapes_dur=True, tracking="all"); print("Got m4")
ea.add_condition("m6", maze_design=6, lights=None, escapes_dur=True, tracking="all"); print("Got m6")
ea.add_condition("m1-light", maze_design=1, lights=1, escapes_dur=True, tracking="all"); print("Got m1-light")
ea.add_condition("m1-dark", maze_design=1, lights=0, escapes_dur=True, tracking="all"); print("Got m1-dark")


ea.add_condition("twolong", maze_design=None, lights=None, escapes_dur=True, tracking="all", experiment_name="TwoArmsLong Maze"); print("Got TwoArmsLong Maze")
ea.add_condition("ff", maze_design=None, lights=None, escapes_dur=True, tracking="all", experiment_name="FlipFlop Maze"); print("Got FlipFlop Maze")
ea.add_condition("ff2", maze_design=None, lights=None, escapes_dur=True, tracking="all", experiment_name="FlipFlop2 Maze"); print("Got FlipFlop2 Maze")
ea.add_condition("fourlong", maze_design=None, lights=None, escapes_dur=True, tracking="all", experiment_name="FourArms Maze"); print("Got FourArms Maze")
ea.add_condition("mb", maze_design=None, lights=None, escapes_dur=True, tracking="all", experiment_name="Model Based"); print("Got Model Based")
ea.add_condition("mb2", maze_design=None, lights=None, escapes_dur=True, tracking="all", experiment_name="Model Based V2"); print("Got Model Based V2")


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
# By arm tracking inspection
arm = 'left'
f, axarr = create_figure(subplots=True, ncols=int(np.ceil(len(ea.conditions.keys())/2)), nrows=2)
for n, (condition, trials) in enumerate(ea.conditions.items()):
    trials = trials.loc[trials.escape_arm == arm]
    for i, trial in trials.iterrows():
        # if arm == 'right':
        #     if np.min(trial.body_xy[:, 0] > 400): continue
        if arm == 'left':
            if np.max(trial.body_xy[:, 0] < 400): continue
        axarr[n].plot(trial.body_xy[:, 0], trial.body_xy[:, 1], color=arms_colors[trial.escape_arm])
    axarr[n].set(title=condition, xlim=[0, 1000], ylim=[0, 1000])

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
n_samples = 250

euclidean_dists = {}
f, axarr = create_figure(subplots=True, nrows=3, ncols=2, sharex=False)
for n, (condition, trials) in enumerate(ea.conditions.items()):
    # Get data and sample aligned in time normalized bu escape duration
    if condition not in five_mazes: continue
    X = np.linspace(0, 1, num=101)
    data = {a:{round(m,2):[] for m in X} for a in arms}
    counts = {a:{round(m,2):0 for m in X} for a in arms}

    for i, trial in trials.iterrows():
        n_frames = trial.body_xy.shape[0] - (trial.out_of_t_frame-trial.stim_frame)

        x = np.linspace(0, 1, num=n_frames)
        d = calc_distance_from_shelter(trial.body_xy[trial.out_of_t_frame-trial.stim_frame:, :], [500, 850])

        for xx, dd in zip(x, d):
            data[trial.escape_arm][round(xx, 2)].append(dd)
            counts[trial.escape_arm][round(xx, 2)] += 1

    # Plot the average from all trials
    for arm in arms:
        if condition != "m0" and arm == "center": continue
        avg = np.array([np.mean(d) if c > 0 else np.nan for d,c in zip(data[arm].values(), counts[arm].values())])
        yerr = np.array([stats.sem(d) if c > 0 else np.nan for d,c in zip(data[arm].values(), counts[arm].values())])

        axarr[n].fill_between(X, avg-yerr, avg+yerr, color=arms_colors[arm], alpha=.4)
        axarr[n].plot(X, avg, color=desaturate_color(arms_colors[arm]), lw=4, label=arm)

    avg_r = np.array([np.mean(d) if c > 0 else np.nan for d,c in zip(data['right'].values(), counts['right'].values())])
    avg_l = np.array([np.mean(d) if c > 0 else np.nan for d,c in zip(data['left'].values(), counts['left'].values())])
    ratio = avg_l/avg_r
    axarr[0].plot(ratio, color=maze_colors[condition], lw=3, label=condition)

    top, bottom = np.max(ratio-1), np.min(ratio-1)
    if top > np.abs(bottom): dist = round(top, 4)
    else: dist = round(bottom, 4)
    print("{} - A.O.C.: {} - max dist raito: {} - max dist L: {} - max dist R {}".format(
            condition, round(np.trapz(ratio-1), 3), dist, round(np.max(avg_l), 3), round(np.max(avg_r), 3)))
    euclidean_dists[condition] = dist
    
    axarr[n].legend()
    axarr[n].set(title=condition, ylim=[100, 650], xticks=[0, 1/2, 1], xlabel='escape proportion', 
                ylabel='eucl.dist.shelt.')

axarr[0].legend(fontsize=9)
axarr[0].set(title="LEFT/RIGHT", xticklabels=[0, 1/2, 1], xlabel='escape proportion', 
            ylabel='L/R', xticks=[0, len(avg_r)*0.5, len(avg_r)])
f.tight_layout()

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
# ---------------------------------------------------------------------------- #
#                                ! PSYCHOMETRIC                                #
# ---------------------------------------------------------------------------- #

includem0 = False
includem6 = True
fit_curve = True

use_eucl = False

alpha, beta = 2.5, 1
combined_dists = {a:(alpha*mazes[a]['ratio'])+(beta*euclidean_dists[a]) for a in euclidean_dists.keys()}

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
        x = mazes[pr.condition]['ratio']
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
    xfit.append(x); yfit.append(pr['mean']); stdfit.append(math.sqrt(pr.sigmasquared))

    ax.errorbar(x, pr['mean'], yerr=math.sqrt(pr.sigmasquared), fmt = 'o', color=maze_colors['m6'])
    plot_distribution(pr.alpha.values[0], pr.beta.values[0], ax=ax, dist_type="beta", shaded="True", line_alpha=.3,
                    plot_kwargs={"color":maze_colors['m6']}, shade_alpha=.05,
                    vertical=True, fill_offset=(x), y_scale=.008)



_ = ax.axhline(0.5, ls="--", color=[.5, .5, .5])

if not use_eucl: 
    xlim=[.5, 2.8]
else:
    xlim=[min(combined_dists.values())-1,  max(combined_dists.values())+1]

if fit_curve:
    plot_fitted_curve(centered_logistic, xfit, yfit, ax, xrange=xlim, scatter_kwargs=dict(alpha=0),
                    fit_kwargs = dict(sigma=stdfit),
                    line_kwargs=dict(color=[.3, .3, .3], alpha=.7, lw=3))


_ = ax.set(title="p(R) for each maze", xticks=X, xlabel="Llong/Short length ratio", 
                    xticklabels=Xlabels, 
                    ylabel="p(best)",
                    ylim=[0, 1], xlim=xlim ) #xlim=[.5, 2.8]
save_plot("psychometric", f)



# %%

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
        if condition not in psychometric_mazes: continue
        n = psychometric_mazes.index(condition)
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
        ax.set(title="{} - {}s window".format(condition, windows_size), xlabel="time (s)", ylabel="p(R)", xlim=[0, 60*60*1.5],
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
        if condition not in psychometric_mazes: continue
        n = psychometric_mazes.index(condition)

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
# TODO plot against null posterior instead of against M4
pRs = ea.bayes_by_condition_analytical()

conditions = ['m4', 'm6']
f, axarr = create_figure(subplots=True, ncols=2)
for i, condition in enumerate(conditions):
    pr = pRs.loc[pRs.condition==condition]

    axarr[0].errorbar(pr['mean'], 0*0.5*i, xerr=math.sqrt(pr.sigmasquared), color=maze_colors[condition])
    axarr[0].scatter(pr['mean'], 0*0.5*i, color=maze_colors[condition], s=250, edgecolor=black, zorder=99, label=condition)


    plot_distribution(pr.alpha.values[0], pr.beta.values[0], ax=axarr[0], dist_type="beta", shaded="True", line_alpha=.3,
                    plot_kwargs={"color":maze_colors[condition]}, shade_alpha=.05, 
                    y_scale=.01)

lengths = [np.sum(calc_distance_between_points_in_a_vector_2d(t.body_xy)) for i,t in ea.conditions['m6'].iterrows()]
llengths = [l  for l, (i, t) in zip(lengths, ea.conditions['m6'].iterrows()) if t.escape_arm == 'left']
elengths = [l  for l, (i, t) in zip(lengths, ea.conditions['m6'].iterrows()) if t.escape_arm == 'right']

axarr[1].hist(lengths, color=maze_colors['m6'], density=True)
axarr[1].hist(llengths, color=green, alpha=.5, density=True)
axarr[1].hist(elengths, color=blue, alpha=.5, density=True)

axarr[0].legend()
_ = axarr[0].axvline(.5, color=black, lw=2, ls=":")
_ = axarr[0].set(title="M4 vs M6", xlabel="p(R)", ylabel="density", xlim=[0, 1])
save_plot("m6", f)


# %%
# ---------------------------------------------------------------------------- #
#                                     ! M0                                     #
# ---------------------------------------------------------------------------- #

trials = ea.conditions['m0']

f, ax = create_figure(subplots=False)

for exclude, shortest, longest in zip(['left', 'right', 'center'], ['center', 'center', 'right'], ['right', 'left', 'left']):
    tleft = trials.loc[trials.escape_arm != exclude]
    pshort = len(tleft.loc[tleft.escape_arm==shortest])/len(tleft)
    lenratio = mazes['m0']["{}_path_length".format(longest)]/mazes['m0']["{}_path_length".format(shortest)]
    print('Excluding {} - length ratio {} - p(shortest) {} - n trials {}'.format(exclude, round(lenratio, 2), round(pshort, 2), len(tleft)))

    color = m0cols[exclude]
    binary_trials = [1 if t.escape_arm == shortest else 0 for i,t in tleft.iterrows()]
    a2, b2, mean, mode, sigmasquared, prange = ea.grouped_bayes_analytical([len(binary_trials)], [np.sum(binary_trials)])
    ax.errorbar(lenratio, mean, yerr=math.sqrt(sigmasquared), fmt = 'o', color=color)

    plot_distribution(a2, b2, ax=ax, dist_type="beta", shaded="True", line_alpha=.3,
                    plot_kwargs={"color":color}, shade_alpha=.05,
                    vertical=True, fill_offset=(lenratio), y_scale=.008)

_ = ax.set(ylim=[0, 1], xlim=[.9, 2.8])

save_plot("m0", f)





# %%
# ---------------------------------------------------------------------------- #
#                                  OTHER MAZES                                 #
# ---------------------------------------------------------------------------- #
for m in ['mb2', 'mb']:
    trials = ea.conditions[m]
    p = [len(trials.loc[trials.escape_arm == a])/len(trials) for a in ['left', 'center', 'right']]
    print(m, " {0} trials, -  L:{1:.3f}, - C:{2:.3f}, - R:{3:.3f}".format(len(trials), *p))

























# %%
# ---------------------------------------------------------------------------- #
#                       OTHER THINGS NOT USED FREQUENTLY                       #
# ---------------------------------------------------------------------------- #

# ---------------------------- ! EFFECT OF ORIGIN ---------------------------- #
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
save_plot("effect_origin", f)


# %%
# -------------------------- ! EXPLORATION ANALYSIS -------------------------- #
exploration_data, alldata = ea.get_exploration_per_path_from_trials()
f, ax = create_figure(subplots=False)
for condition, data in exploration_data.items():
    if condition == "m0": continue
    x = [i-.25, i+.25]
    xticks.extend([i-.25,i, i+.25])
    xlabels.extend(["left","\n{}".format(condition), "right"])
    y = [data.left.median/mazes[condition]['ratio'], data.right.median]
    yerr = [data.left.std, data.right.std]

    ax.axvline(i-.25, color=[.2, .2, .2], ls=":", alpha=.15)
    ax.axvline(i+.25, color=[.2, .2, .2], ls=":", alpha=.15)

    ax.plot(x, y, "-o", label=condition, color=maze_colors[condition], zorder=90)
    ax.scatter(x, y, edgecolor=black, s=250, color=maze_colors[condition], zorder=99)
    ax.errorbar(x, y, yerr, color=maze_colors[condition], zorder=90)


    ttest, pval = stats.ttest_ind([x/mazes[condition]['ratio'] for x in  alldata[condition]['left']], 
                                    alldata[condition]['right'])
    if pval < .05:
        ax.plot([i-.3, i+.3], [4.25, 4.25], lw=4, color=[.4, .4, .4])
        ax.text(i-0.025, 4.25, "*", fontsize=20)
    else:
        ax.plot([i-.3, i+.3], [4.25, 4.25], lw=4, color=[.7, .7, .7])
        ax.text(i-0.025, 4.25, "n.s.", fontsize=16)

    x = [0, 1]
    y = [data.left.median/mazes[condition]['ratio'], 
        data.right.median]

    ax.plot(x, y, color=maze_colors[condition])
    ax.scatter(x, y, s=250, edgecolor=black, color=maze_colors[condition], label=condition)

ax.legend()
_ = ax.set(title="Normalized arm occupancy during exploration", xlabel="arm", xticks=[0, 1], 
        xticklabels=["left", "right"], ylabel="norm. occupancy (s/len)")
save_plot("effect_exploration", f)



# %%
# ----------------------------- ! LIGHT VS DARK ----------------------------- #
# Light vs dark -- get data
ea2 = ExperimentsAnalyser(load_psychometric=False, tracking="all")
ea2.max_duration_th = 9

ea2.add_condition("m1-light", maze_design=1, lights=1, escapes_dur=True, tracking="all")
ea2.add_condition("m1-dark", maze_design=1, lights=0, escapes_dur=True, tracking="all")

for condition, trials in ea2.conditions.items():
    print("Maze {} -- {} trials".format(condition, len(trials)))


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
