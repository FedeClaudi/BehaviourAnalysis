import sys
sys.path.append('./')

from Utilities.imports import *
from Analysis.Behaviour.utils.experiments_analyser import ExperimentsAnalyser
from Analysis.Behaviour.utils.T_utils import get_T_data, get_angles, get_above_yth
%matplotlib inline

aligned_trials  = get_T_data(load=False, median_filter=True)


# Plot polar histogram of orientation at yth
f = plt.figure(figsize=(8, 8), facecolor=white)

yths = [200, 225, 250, 275, 300, 325]


gs = gridspec.GridSpec(len(yths), 3) 
tracking_ax = plt.subplot(gs[:, :2])
polaxs = [plt.subplot(gs[i, 2], projection='polar') for i in range(len(yths))]


colors = [lilla, magenta, orange, green, lightblue, red]

thetas = {t:[] for t in yths}
for counter, (i, trial) in tqdm(enumerate(aligned_trials.iterrows())):
    # Get right aligned tracking data
    x, y, s = trial.tracking[:, 0], trial.tracking[:, 1], trial.tracking[:, 2]
    sx, nx, tx  = trial.s_tracking[:, 0], trial.n_tracking[:, 0], trial.t_tracking[:,0]
    sy, ny, ty  = trial.s_tracking[:, 1], trial.n_tracking[:, 1], trial.t_tracking[:,1]

    if trial.escape_side == "left":
        x, sx, tx, ns = 500 + (500 - x), 500 + (500 - sx), 500 + (500 - tx), 500 + (500 - nx)

    # Get angle when mouse goes over the y threshold (for each th)
    for yth in yths:
        below_yth, above_yth = get_above_yth(y, yth)
        try:
            angle = trial.body_orientation[above_yth]
        except: continue
        if angle < 5 or angle > 355: continue
        thetas[yth].append(np.radians(angle))
    tracking_ax.plot(x, y, color=grey, alpha=.6, lw=2)

# Rose plots
for polax, yth, color in zip(polaxs[::-1], yths, colors):
    rose_plot(polax, np.array(thetas[yth]), edge_color=white, bins=20, xticks=False, linewidth=2)
    polax.plot([average_angles(thetas[yth]), average_angles(thetas[yth])], [0, 0.5], color=color, lw=4)
    _ = polax.set(title="Yth: {}".format(yth))

# Add stuff to plots
[tracking_ax.axhline(yth, color=color, lw=2) for (yth, color) in zip(yths, colors)]

# set axes
_  = tracking_ax.set(facecolor=[.2, .2, .2], ylim=[120, 370], xlim=[425, 575])
for ax in polaxs: 
    ax.set(facecolor=[.2, .2, .2])

f.tight_layout()
