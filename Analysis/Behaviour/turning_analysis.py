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
ea = ExperimentsAnalyser(load=False,  naive=None, lights=1, escapes=True, escapes_dur=True,  shelter=True)

# Get Threat Platform data
ea.prep_tplatf_trials_data(filt=False, remove_errors=True, speed_th=None)

# Get Torosity Utils
tor = Torosity()

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
# ! PLot a bunch of KDE for all trials
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
# ! Plot trackings
f, axarr = create_figure(subplots=True, ncols=3, facecolor=white, figsize=(15,15))

for i, trial in aligned_trials.iterrows():
    axarr[0].scatter(trial.tracking[trial.turning_frame, 0], trial.tracking[trial.turning_frame, 1], color=red, alpha=.8, zorder=99)
    axarr[0].plot(trial.tracking[:, 0], trial.tracking[:, 1], color=white, alpha=.2, lw=3)

    axarr[1].scatter(trial.tracking_centered[trial.turning_frame, 0], trial.tracking_centered[trial.turning_frame, 1], color=red, alpha=.8, zorder=99)
    axarr[1].plot(trial.tracking_centered[:, 0], trial.tracking_centered[:, 1], color=white, alpha=.2, lw=3)

    axarr[2].scatter(trial.tracking_turn[trial.turning_frame, 0], trial.tracking_turn[trial.turning_frame, 1], color=red, alpha=.8, zorder=99)
    axarr[2].plot(trial.tracking_turn[:, 0], trial.tracking_turn[:, 1], color=white, alpha=.2, lw=3)


#%%
# ? WIP
# Configure Plotly to be rendered inline in the notebook.
plotly.offline.init_notebook_mode()


yth = 200

colorhelper = MplColorHelper("viridis", 0, 2, rgb255=True)

fig = go.Figure()
plot_data = []
for i, trial in aligned_trials.iterrows():
    try:
        below_yth = np.where(trial.tracking[:, 1] <= yth)[0][-1]
    except: 
        above_yth = 0
    else:
        above_yth = below_yth + 1

    # small_tracking = tor.smallify_tracking(trial.tracking[above_yth:, :])

    # ax.plot(trial.tracking[above_yth:, 0], trial.tracking[above_yth:, 1], trial.tracking[above_yth:, 2], lw=2)
    
    color = colorhelper.get_rgb(np.nanmedian(trial.tracking[:, 2]))
    # Configure the trace.
    trace = go.Scatter3d(
        x=trial.tracking[above_yth:, 0],  # <-- Put your data instead
        y=trial.tracking[above_yth:, 1],  # <-- Put your data instead
        z=trial.tracking[above_yth:, 2],  # <-- Put your data instead
        # mode='markers',
        marker={
            'size': 1,
            'opacity': 0.8,
        },
        line=dict(
        color = [color for i in range(len(trial.tracking[above_yth:, 0]))],
        width=4
        )
    )
    plot_data.append(trace)

    # trial_torosity = tor.process_one_trial(tracking=np.int32(small_tracking))

    # if i > 20: break


fig = go.Figure(data=plot_data, layout=layout)

camera = dict(
    eye=dict(x=0., y=0., z=2.5)
)
scene =dict(
            xaxis = dict(
                    backgroundcolor="rgb(55, 55, 55)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",),
            yaxis = dict(
                backgroundcolor="rgb(55, 55, 55)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white"),
            zaxis = dict(
                backgroundcolor="rgb(55, 55, 55)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                range=[0, 10]),
            xaxis_title='X AXIS TITLE',
            yaxis_title='Y AXIS TITLE',
            zaxis_title='Z AXIS TITLE',
            )

fig.update_layout(scene = scene, scene_camera=camera,
                    width=700,
                    margin=dict(r=10, l=10,b=10, t=10),
                  )

plotly.offline.iplot(fig)

# ax.axhline(yth*tor.scale_factor, color=white, lw=2)
# ax.set(facecolor=[.2, .2, .2],  xlim=[400, 600], ylim=[120, 375])




#%%
f, ax = create_figure(subplots=False, facecolor=white, figsize=(15,15))
yth = 200
sth, high_sth =  4, 5

median_speed, max_speed = [], []
for i, trial in aligned_trials.iterrows():
    if "left" == trial.escape_side:
        x, y, s = 500 + (500 - trial.tracking[:, 0]), trial.tracking[:, 1], trial.tracking[:, 2]
    else:
        x, y, s = trial.tracking[:, 0], trial.tracking[:, 1], trial.tracking[:, 2]

    try:
        below_yth = np.where(y <= yth)[0][-1]
    except: 
        above_yth = 0
    else:
        above_yth = below_yth + 1

    slow_idx = np.where(s[below_yth:] < sth)[0]+below_yth
    fast_idx = np.where(s[below_yth:] >= high_sth)[0]+below_yth
    median_speed.append(np.nanmedian(s))
    max_speed.append(np.max(s))

    ax.plot(x[:above_yth], y[:above_yth], color=[.6, .6, .6], lw=1)
    ax.plot(x[above_yth:], y[above_yth:], color=white, lw=1)
    ax.scatter(x[slow_idx], y[slow_idx], color=red, s=30, zorder=99, alpha=.5)
    ax.scatter(x[fast_idx], y[fast_idx], color=green, s=30, zorder=99, alpha=.5)

    # if i > 20: break


ax.axhline(yth, color=white, lw=2)
_ = ax.set(title="red=slow, green=fast", facecolor=[.2, .2, .2],  xlim=[400, 600], ylim=[120, 375])


kdeax, _, = plot_kde(data=median_speed, kde_kwargs={"bw":.25}, fig_kwargs={'facecolor':white},  color=red)
kdeax, _, = plot_kde(ax=kdeax, data=max_speed, kde_kwargs={"bw":.25}, color=green)

kdeax.axvline(sth, color=red)
kdeax.axvline(high_sth, color=green)

_ = kdeax.set(title="KDE of median speed", facecolor=[.2, .2, .2], xlim=[0, 15])


#%%
