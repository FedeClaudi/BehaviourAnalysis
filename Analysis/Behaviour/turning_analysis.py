# %%
import sys
sys.path.append('./')

from Utilities.imports import *
from Analysis.Behaviour.experiments_analyser import ExperimentsAnalyser
from Analysis.Behaviour.torosity import Torosity

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks

import plotly
import plotly.graph_objs as go
import colorlover as cl

%matplotlib inline

# %%
# Getting data
ea = ExperimentsAnalyser(load=False,  naive=None, lights=1, escapes=True, escapes_dur=True,  shelter=True, 
                    agent_params={'kernel_size':3, 'model_path':'PathInt2_old.png', 'grid_size':1000})

# Get Threat Platform data
ea.prep_tplatf_trials_data(filt=False, remove_errors=True, speed_th=None)

# # Get Torosity Utils
# tor = Torosity()

#%%
# ! Get aligned tracking data
xth, yth, thetath = 10, 50, 5
upfps = 1000

delta_x_f, delta_y_f = [], []
cmaps = ["", "Greens", "Blues", ""]

ch = MplColorHelper("Greens", 0, 150)

aligned_trials = {'trial_id':[], "tracking":[], "tracking_centered":[], "tracking_turn":[], "turning_frame":[], 
                    "escape_side":[]}

for i, (condition, trials) in enumerate(ea.trials.items()):
    if condition == "maze1" or condition == "maze4": continue  # Only work on trials with the catwalk

    # loop over trials
    for n, (ii, trial) in enumerate(trials.iterrows()):
        # Get body, snout tail tracking
        rawx, rawy = trial.threat_tracking[:, 0], trial.threat_tracking[:, 1]
        x, y = trial.threat_tracking[:, 0]-trial.threat_tracking[0, 0], trial.threat_tracking[:, 1]-trial.threat_tracking[0, 1]
        s = trial.threat_tracking[:, 2]
        tx, ty = trial.tail_threat_tracking[:, 0]-trial.threat_tracking[0, 0], trial.tail_threat_tracking[:, 1]-trial.threat_tracking[0, 1]
        sx, sy = trial.snout_threat_tracking[:, 0]-trial.threat_tracking[0, 0], trial.snout_threat_tracking[:, 1]-trial.threat_tracking[0, 1]

        # If escaping on the left, flipt the x tracking
        if "left" in trial.escape_arm.lower():  
            x -= x
            sx -= sx
            tx -= tx
            left = True
        else:
            left = False

        # get time
        t = np.arange(len(x))
        upt = upsample_signal(40, upfps, t)

        # Get angular velocity
        body_angles = calc_angle_between_vectors_of_points_2d(np.vstack([x, y]), np.vstack([tx, ty]))
        head_angles = calc_angle_between_vectors_of_points_2d(np.vstack([sx, sy]), np.vstack([x, y]))
        avg_angles = np.mean(np.vstack([body_angles, head_angles]), axis=0)
        dir_of_mvt = calc_angle_between_points_of_vector(np.vstack([x, y]).T)

        # get point of max distance from straightlinr
        dist = [calc_distance_between_point_and_line([[x[0], y[0]], [x[-1], y[-1]]], [_x, _y]) 
                        for _x,_y in zip(x, y)]
        mdist = np.argmax(dist)

        aligned_trials['trial_id'].append(trial['trial_id'])
        aligned_trials['tracking'].append(np.vstack([rawx, rawy, s]).T)
        aligned_trials['tracking_centered'].append(np.vstack([x, y, s]).T)
        aligned_trials['tracking_turn'].append(np.vstack([x-x[mdist], y-y[mdist], s]).T)
        aligned_trials['turning_frame'].append(mdist)
        
        if left:  
            aligned_trials['escape_side'].append("left")
        else:
            aligned_trials['escape_side'].append("right")

aligned_trials = pd.DataFrame.from_dict(aligned_trials)
aligned_trials


#%%
# Plot tracking and highlight fast and slow frames
f, axarr = create_triplot(facecolor=white, figsize=(15,15))

# thresholds
yth = 200
sth, high_sth =  2, 4
cdth = 100

median_speed, max_speed, all_speeds, cum_dist_from_line_l = [], [], [], []
for i, trial in aligned_trials.iterrows():
    # Get right aligned tracking data
    if "left" == trial.escape_side:
        x, y, s = 500 + (500 - trial.tracking[:, 0]), trial.tracking[:, 1], trial.tracking[:, 2]
        x_cent, y_cent = 500 + (500 - trial.tracking_centered[:, 0]), trial.tracking_centered[:, 1]
    else:
        x, y, s = trial.tracking[:, 0], trial.tracking[:, 1], trial.tracking[:, 2]
        x_cent, y_cent = trial.tracking_centered[:, 0], trial.tracking_centered[:, 1]

    # Get when mouse goes over the y threshold
    try:
        below_yth = np.where(y <= yth)[0][-1]
    except: 
        below_yth = 0
        above_yth = 0
    else:
        above_yth = below_yth + 1

    # get fast and slow frames
    slow_idx = np.where(s[below_yth:] < sth)[0]+below_yth
    fast_idx = np.where(s[below_yth:] >= high_sth)[0]+below_yth
    median_speed.append(np.nanmedian(s))
    max_speed.append(np.max(s))
    all_speeds.extend([x for x in s if not np.isnan(x)])

    # Get comulative distance between tracking and a straight line going
    # between the mouse location when it gets above yth and when it leaves T
    p0, p1 = [x[above_yth], y[above_yth]], [x[-1], y[-1]]
    cumdist = np.sum(np.array(cals_distance_between_vector_and_line([p0, p1], np.vstack([x[above_yth:], y[above_yth:]]).T))**2)/len(x[above_yth:])
    cum_dist_from_line_l.append(cumdist)


    # plot tracking
    if cumdist < cdth:
        color = red
    else:
        color = green

    axarr.main.plot(x[:above_yth], y[:above_yth], color=desaturate_color(color), lw=1)
    axarr.main.plot(x[above_yth:], y[above_yth:], color=color, lw=1)

    # Highlight fast and slow frames
    # axarr.main.scatter(x[slow_idx], y[slow_idx], color=red, s=30, zorder=99, alpha=.5)
    # axarr.main.scatter(x[fast_idx], y[fast_idx], color=green, s=30, zorder=99, alpha=.5)

# Plot kde of speeds distributions
kdeax, _, = plot_kde(data=all_speeds, kde_kwargs={"bw":.1}, fig_kwargs={'facecolor':white},  color=white)
kdeax.axvline(sth, color=red)
kdeax.axvline(high_sth, color=green)

# plot KDE of comulative line distances
_, _, = plot_kde(data=cum_dist_from_line_l, ax=axarr.x, kde_kwargs={"bw":10}, color=white)
axarr.x.axvline(cdth, color=green)

# Set axes props
axarr.main.axhline(yth, color=white, lw=2)
_ = axarr.main.set(title="red=slow, green=fast", facecolor=[.2, .2, .2],  ylim=[120, 370], xlim=[475, 575]) # 
_ = kdeax.set(title="KDE of median speed", facecolor=[.2, .2, .2], xlim=[0, 15])
_= axarr.x.set(title="comulative line distance", facecolor=[.2, .2, .2], xlim=[-500, 750])


#%%
