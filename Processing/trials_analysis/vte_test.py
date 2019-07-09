# %%
import sys
sys.path.append('./')

from Utilities.imports import *
from mpl_toolkits.mplot3d import Axes3D

from Processing.trials_analysis.all_trials_loader import Trials

%matplotlib inline  
# %%
# Get data
stimuli = (Session & "experiment_name='PathInt2'") * Stimuli 



# %%
# Arrify data
n_frames = 120
unity = [[-5, -5], [25, 25]]

data = np.zeros((n_frames, 3, len(stimuli)))
for i, stim in enumerate(stimuli):
    if  stim["duration"] < 0: continue
    
    # Get body and snout trackign data
    start = stim["overview_frame"]
    end = start + n_frames

    snout = (TrackingData.BodyPartData & "bpname='snout'" & "recording_uid='{}'".format(stim["recording_uid"])).fetch1("tracking_data")[start:end, :]
    body = (TrackingData.BodyPartData & "bpname='body'" & "recording_uid='{}'".format(stim["recording_uid"])).fetch1("tracking_data")[start:end, :]
    tail = (TrackingData.BodyPartData & "bpname='tail_base'" & "recording_uid='{}'".format(stim["recording_uid"])).fetch1("tracking_data")[start:end, :]

    s_speed, b_speed = line_smoother(snout[:,2],  window_size=11, order=3,), line_smoother(body[:, 2],  window_size=11, order=3,)
    ratio = s_speed / b_speed
    data[:len(b_speed), 0, i] = b_speed
    data[:len(b_speed), 1, i] = s_speed
    data[:len(b_speed), 2, i] = ratio

    # Plot the tracking tracjectory and the snout over body velocity and save as a figure
    fig, axarr = plt.subplots(ncols=3, figsize=(16, 12))

    # Plot snout over body velocity
    axarr[0].scatter(b_speed, s_speed, cmap="Reds", c=np.arange(len(s_speed)), label="time", alpha=.5)
    axarr[0].plot(b_speed, s_speed, color=[.8, .8, .8], lw=1, alpha=.15)

    axarr[0].plot([0, 25], [0, 25], color="w", lw=2, alpha=.2)
    axarr[0].legend()
    axarr[0].set(title="snout over body speed", xlabel="body speed", ylabel="snout speed", facecolor=[.1, .1, .1])

    # Plot distance from unity line
    d = [calc_distane_between_point_and_line(unity, data[n, :2, i]) for n in np.arange(n_frames)]
    max_d_frame, min_d_frame = np.argmax(d), np.argmin(d)

    axarr[2].plot(d, color="r", lw=2)
    axarr[2].scatter(max_d_frame, d[max_d_frame], color="m", s=100)
    axarr[2].scatter(min_d_frame, d[min_d_frame], color="orange", s=100)
    axarr[2].set(title="Distance from unity", ylabel="distance", xlabel="frames", facecolor=[.1, .1, .1])
    

    axarr[0].scatter(b_speed[max_d_frame], s_speed[max_d_frame], color="m", s=200, alpha=.5)
    axarr[0].scatter(b_speed[min_d_frame], s_speed[min_d_frame], color="orange", s=200, alpha=.5)

    # Plot tracking data
    for f in np.arange(len(b_speed)):
        if f % 10 == 0: alpha=1
        else: alpha= .1

        # alpha=normalise_1d(d)[f][0]

        axarr[1].plot([snout[f, 0], body[f, 0]], [snout[f, 1], body[f, 1]], color="g", alpha=alpha)
        axarr[1].plot([tail[f, 0], body[f, 0]], [tail[f, 1], body[f, 1]], color="b", alpha=alpha)
        axarr[1].scatter(snout[f, 0], snout[f, 1], color="r", alpha=alpha)
        axarr[1].scatter(body[f, 0], body[f, 1], color="g", alpha=alpha)
        axarr[1].scatter(tail[f, 0], tail[f, 1], color="b", alpha=alpha)

        if f == max_d_frame:
            axarr[1].scatter(snout[f, 0], snout[f, 1], color="m", s=400, alpha=.75)
        elif f  == min_d_frame:
            axarr[1].scatter(snout[f, 0], snout[f, 1], color="orange", s=400, alpha=.75)


    axarr[1].set(title="title tracking trajectory", xlabel="x pos", ylabel="y pos", facecolor=[.1, .1, .1], xlim=[150, 850], ylim=[200, 800])

    save_figure(fig, "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\vte_test\\{}.png".format(i))
    close_figure(fig)
    # break


 # %%
# Get distance from unity line over time
distances = np.zeros((n_frames, data.shape[2]))
for i in np.arange(data.shape[2]):
    d = [calc_distane_between_point_and_line(unity, data[n, :2, i]) for n in np.arange(n_frames)]
    distances[:, i] = d

# %%
# Align by peak location
# aligned = np.zeros((240, distances.shape[1]))
peaks = np.argmax(distances, 0)
# for i, p in enumerate(peaks):
#     shifted = np.hstack([np.zeros(p), distances[:, i]])
#     aligned[:, i] = shifted


# Plot distances
f, ax = plt.subplots()
sort_idx = np.argsort(np.argmax(distances, 1))
for i, p in enumerate(peaks):
    x = np.linspace(-p, n_frames-p, n_frames)
    ax.plot(x, distances, color="k", alpha=.5)
ax.plot(np.mean(aligned, 1), color="r", alpha=1)


#%%
# TODO look at snout speed vs body speed for each trial

f, ax = plt.subplots()

for i in tqdm( range(data.shape[2])):
    if i > 20 and i < 30:
        ax.plot(data[:, 0, i], data[:, 1, i], alpha=.6)
    
ax.plot([0, 25], [0, 25], color="r", lw=2)
ax.set(xlabel="body speed", ylabel="Snout speed")

# %%
# Plot just ratio
f, ax = plt.subplots()

for i in tqdm( range(data.shape[2])):
    ax.plot(data[:, 2, i], alpha=.5)

ax.set(xlabel="frames", ylabel="snout/body ratio", xlim=[0, 90], ylim=[-10, 10], facecolor=[.2, .2, .2])





#%%
