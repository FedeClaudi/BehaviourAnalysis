# %%
import sys
sys.path.append('./')

from Utilities.imports import *
from Analysis.Behaviour.experiments_analyser import ExperimentsAnalyser
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import find_peaks


"""
    For now this script finds the "bending" point of the threat traces by drawing a straight line between
    the mouse position at stim onset and where it leaves T. The bending point then is the point of the tracking
    trace at the highest distance from theat line. 
"""

# TODO clean up tracking errors
# TODO Find way to store aligned tracking data and before/after turn data better
# TODO Get speed before and after turn data
# TODO Heatmap of turning locations

# %%
ea = ExperimentsAnalyser(load=False,  naive=None, lights=1, escapes=True, escapes_dur=True)
# Get Threat Platform data
ea.prep_tplatf_trials_data(filt=False, remove_errors=True, speed_th=None)
%matplotlib inline


#%%
# Plot centered tracking traces
f, axarr = create_figure(subplots=True, ncols=3, facecolor=white, figsize=(24,24))

xth, yth, thetath = 10, 50, 5
upfps = 1000

delta_x_f, delta_y_f = [], []
cmaps = ["", "Greens", "Blues", ""]

ch = MplColorHelper("Greens", 0, 150)

for i, (condition, trials) in enumerate(ea.trials.items()):
    if condition == "maze1" or condition == "maze4": continue  # Only work on trials with the catwalk

    # loop over trials
    for n, (ii, trial) in enumerate(trials.iterrows()):
        # Get body, snout tail tracking
        rawx, rawy = trial.threat_tracking[:, 0], trial.threat_tracking[:, 1]
        x, y = trial.threat_tracking[:, 0]-trial.threat_tracking[0, 0], trial.threat_tracking[:, 1]-trial.threat_tracking[0, 1]
        tx, ty = trial.tail_threat_tracking[:, 0]-trial.threat_tracking[0, 0], trial.tail_threat_tracking[:, 1]-trial.threat_tracking[0, 1]
        sx, sy = trial.snout_threat_tracking[:, 0]-trial.threat_tracking[0, 0], trial.snout_threat_tracking[:, 1]-trial.threat_tracking[0, 1]

        # If escaping on the left, flipt the x tracking
        if "left" in trial.escape_arm.lower():  
            x -= x
            sx -= sx
            tx -= tx

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

        # plot
        # plot tracking
        axarr[0].plot(rawx, rawy, color=ch.get_rgb(n),  alpha=.4)
        axarr[1].plot(x, y, color=ch.get_rgb(n),  alpha=.4)

        axarr[2].plot(x[:mdist]-x[mdist], y[:mdist]-y[mdist], color=red,  alpha=.4)
        axarr[2].plot(x[mdist:]-x[mdist], y[mdist:]-y[mdist], color=white,  alpha=.4)
    

for ax, ttl in zip(axarr, ['tracking', 'aligned-start', 'aligned-turn']):
    ax.set(title=ttl, facecolor=[.2, .2, .2])





#%%
