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

        # if mdist == 0: 
        #     print("0")
        #     continue
        # if not np.any(~np.isnan(x-x[mdist])): 
        #     print("cacca")
        #     continue

        # plot
        # plot tracking
        # axarr[0].plot(rawx, rawy, color=ch.get_rgb(n),  alpha=.4)
        # axarr[1].plot(x, y, color=ch.get_rgb(n),  alpha=.4)

        # axarr[2].plot(x[:mdist]-x[mdist], y[:mdist]-y[mdist], color=red,  alpha=.4)
        # axarr[2].plot(x[mdist:]-x[mdist], y[mdist:]-y[mdist], color=white,  alpha=.4)

        aligned_trials['trial_id'].append(trial['trial_id'])
        aligned_trials['tracking'].append(np.vstack([rawx, rawy, s]).T)
        aligned_trials['tracking_centered'].append(np.vstack([x, y, s]).T)
        aligned_trials['tracking_turn'].append(np.vstack([x-x[mdist], y-y[mdist], s]).T)
        aligned_trials['turning_frame'].append(mdist)
        
        if left:  
            aligned_trials['escape_side'].append("left")
            print("l2")
        else:
            aligned_trials['escape_side'].append("right")

for ax, ttl in zip(axarr, ['tracking', 'aligned-start', 'aligned-turn']):
    ax.set(title=ttl, facecolor=[.2, .2, .2])



aligned_trials = pd.DataFrame.from_dict(aligned_trials)
aligned_trials
#%%
# PLot a bunch of KDE for all trials
f, axarr = create_triplot(facecolor=white, figsize=(15,15))


mean_speed_pre, mean_speed_post, y_at_turn = [], [], []
for i, trial in aligned_trials.iterrows():
    axarr.main.scatter(trial.tracking[trial.turning_frame, 0], trial.tracking[trial.turning_frame, 1], color=red, alpha=.8, zorder=99)
    axarr.main.plot(trial.tracking[:, 0], trial.tracking[:, 1], color=white, alpha=.2, lw=3)
    mean_speed_pre.append(np.mean(trial.tracking[:trial.turning_frame, 2]))
    mean_speed_post.append(np.mean(trial.tracking[trial.turning_frame:, 2]))
    y_at_turn.append((trial.tracking[trial.turning_frame, 1]))


plot_kde(axarr.x, 0, data=mean_speed_pre, color=red, kde_kwargs={"bw":0.25}, label="pre")
plot_kde(axarr.x, 0, data=mean_speed_post, color=white, kde_kwargs={"bw":0.25}, label="post")
plot_kde(axarr.y, 0, data=y_at_turn, color=white, vertical=True, kde_kwargs={"bw":5}, label="post")



axarr.main.set(facecolor=[.2, .2, .2], xlim=[400, 600], ylim=[120, 375])
axarr.x.set(facecolor=[.2, .2, .2])
axarr.y.set(facecolor=[.2, .2, .2], ylim=[120, 375])

axarr.x.legend()
#%%
# Plot KDE for left vs right trials and p(R) vs 
f, axarr = create_triplot(facecolor=white, figsize=(15,15))


mean_speed_pre, mean_speed_post, y_at_turn = {'left':[], 'right':[]},  {'left':[], 'right':[]},  {'left':[], 'right':[]}
for i, trial in aligned_trials.iterrows():
    # axarr.main.scatter(trial.tracking[trial.turning_frame, 0], trial.tracking[trial.turning_frame, 1], color=red, alpha=.8, zorder=99)
    # axarr.main.plot(trial.tracking[:, 0], trial.tracking[:, 1], color=white, alpha=.2, lw=3)

    mean_speed_pre[trial.escape_side].append(np.mean(trial.tracking[:trial.turning_frame, 2]))
    mean_speed_post[trial.escape_side].append(np.mean(trial.tracking[trial.turning_frame:, 2]))
    y_at_turn[trial.escape_side].append((trial.tracking[trial.turning_frame, 1]))


# plot_kde(axarr.x, 0, data=mean_speed_pre, color=red, kde_kwargs={"bw":0.25}, label="pre")
# plot_kde(axarr.x, 0, data=mean_speed_post, color=white, kde_kwargs={"bw":0.25}, label="post")
plot_kde(axarr.y, 0, data=y_at_turn['left'], color=white, vertical=True, kde_kwargs={"bw":5}, label="left")
plot_kde(axarr.y, 0, data=y_at_turn['right'], color=red, vertical=True, kde_kwargs={"bw":5}, label="right")



axarr.main.set(facecolor=[.2, .2, .2], xlim=[400, 600], ylim=[120, 375])
axarr.x.set(facecolor=[.2, .2, .2])
axarr.y.set(facecolor=[.2, .2, .2], ylim=[120, 375])

axarr.x.legend()
axarr.y.legend()

#%%
