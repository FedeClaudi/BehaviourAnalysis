"""
    Analyze the correlation between direction of the first turn and escape path

"""
# %%
import sys
sys.path.append('./')

from Utilities.imports import *
from Analysis.Behaviour.experiments_analyser import ExperimentsAnalyser
from Analysis.Behaviour.T_utils import get_T_data, get_angles, get_above_yth
%matplotlib inline

# Getting data
aligned_trials  = get_T_data(load=True, median_filter=True)

# %%
fig = plt.figure(figsize=(16, 16), facecolor=white)
gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1]) 

tracking_ax = plt.subplot(gs[0, 0])
polax = plt.subplot(gs[0, 1], projection='polar')

# thresholds
yth = 250

for counter, (i, trial) in tqdm(enumerate(aligned_trials.iterrows())):
    if counter == 1: break

    # Get right aligned tracking data
    x, y, s = trial.tracking[:, 0], trial.tracking[:, 1], trial.tracking[:, 2]
    sx, nx, tx  = trial.s_tracking[:, 0], trial.n_tracking[:, 0], trial.t_tracking[:,0]
    sy, ny, ty  = trial.s_tracking[:, 1], trial.n_tracking[:, 1], trial.t_tracking[:,1]

    if trial.escape_side == "left":
        x, sx, tx, ns = 500 + (500 - x), 500 + (500 - sx), 500 + (500 - tx), 500 + (500 - nx)

    # Get when mouse goes over the y threshold
    try:
        below_yth = np.where(y <= yth)[0][-1]
    except: 
        below_yth = 0
        above_yth = 0
    else:
        above_yth = below_yth + 1

    # Get angle between mouse initial position and where it leaves T
    escape_angle = angle_between_points_2d_clockwise([x[0], y[0]], [x[-1], y[-1]])
    # same but for other escape arm
    escape_angle_opposite = angle_between_points_2d_clockwise([x[0], y[0]], [500 + (500 - x)[-1], y[-1]])
    # same but for shelter
    shelter_angle = angle_between_points_2d_clockwise([x[0], y[0]], shelter_location)

    # plot tracking
    frames = np.arange(0, 20, 5)
    tracking_ax.plot([tx, x], [ty, y], color=grey, lw=1, zorder=50)
    tracking_ax.plot([x, nx], [y, ny], color=grey, lw=1, zorder=50)
    tracking_ax.plot([nx, sx], [ny, sy], color=red, lw=1, zorder=50)
    tracking_ax.scatter(sx, sy, color=red, zorder=99, s=15, alpha=.8)

    # plot lines to escape arms and shelter
    tracking_ax.plot([x[0], x[-1]], [y[0], y[-1]], color=green, lw=2)
    tracking_ax.plot([x[0], 500 + (500 - x)[-1]], [y[0], y[-1]], color=lilla, lw=2)
    tracking_ax.plot([x[0], shelter_location[0]], [y[0], shelter_location[1]], color=blue, lw=2)

    # plot angles
    time, maxt = np.arange(len(trial['body_orientation'])), len(trial['body_orientation'])
    polax.plot(np.radians(trial['body_orientation']), time, lw=4, color=white, zorder=99)
    # polax.scatter(np.radians(trial['head_orientation']), time, s=15, color=red, zorder=99)

    # Plot line towards the escape
    polax.plot([np.radians(escape_angle), np.radians(escape_angle)], [0, maxt], lw=2, color=green)
    polax.plot([np.radians(escape_angle+180), np.radians(escape_angle+180)], [0, maxt], lw=2, color=green)

    polax.plot([np.radians(escape_angle_opposite), np.radians(escape_angle_opposite)], [0, maxt], lw=2, color=orange)
    polax.plot([np.radians(escape_angle_opposite+180), np.radians(escape_angle_opposite+180)], [0, maxt], lw=2, color=orange)

    polax.plot([np.radians(shelter_angle), np.radians(shelter_angle)], [0, maxt], lw=2, color=blue)
    polax.plot([np.radians(shelter_angle+180), np.radians(shelter_angle+180)], [0, maxt], lw=2, color=blue)


    # Plot a line representing the animals initial orientation
    polax.plot([np.radians(trial['body_orientation'][0]), np.radians(trial['body_orientation'][0])], [0, 10], lw=8, color=red, zorder=99)

# Set tracking_axes props
tracking_ax.axhline(yth, color=white, lw=2)
_ = tracking_ax.set(title="Tracking. Escape on {}".format(trial.escape_side), facecolor=[.2, .2, .2],  ylim=[200, 500], xlim=[425, 575]) 
_ = polax.set(ylim=[0, above_yth],facecolor=[.2, .2, .2], title="Angle of body over time, below Yth")
polax.grid(False)



# %%
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

# %%
