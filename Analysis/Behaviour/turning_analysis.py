# %%
import sys
sys.path.append('./')

from Utilities.imports import *
from Analysis.Behaviour.experiments_analyser import ExperimentsAnalyser
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import find_peaks

# %%
ea = ExperimentsAnalyser(load=False,  naive=None, lights=1, escapes=True, escapes_dur=True)
# Get Threat Platform data
ea.prep_tplatf_trials_data(filt=False, remove_errors=True,)
%matplotlib inline


#%%
# Plot centered tracking traces
f, axarr = create_triplot(facecolor=white, figsize=(16, 12))
f2, axarr2 = create_figure(subplots=True, ncols=2, facecolor=white)
plt.figure(facecolor=white, figsize=(20, 20))
polax = plt.subplot(111, projection="polar")

xth, yth, thetath = 20, 50, 2.5
upfps = 1000

delta_x_f, delta_y_f = [], []
cmaps = ["", "Greens", "Blues", ""]
for i, (condition, trials) in enumerate(ea.trials.items()):
    if condition == "maze1" or condition == "maze4": continue

    ch = MplColorHelper(cmaps[i], 0,300)

    for n, (ii, trial) in enumerate(trials.iterrows()):
        x, y = trial.threat_tracking[:, 0]-trial.threat_tracking[0, 0], trial.threat_tracking[:, 1]-trial.threat_tracking[0, 1]
        xs, ys = trial.tail_threat_tracking[:, 0]-trial.threat_tracking[0, 0], trial.tail_threat_tracking[:, 1]-trial.threat_tracking[0, 1]
        
        if "left" in trial.escape_arm.lower():  
            x -= x
            xs -= xs

        # Get angular velocity
        angles = calc_angle_between_vectors_of_points_2d(np.vstack([xs, ys]), np.vstack([x, y]))
        upangles = upsample_signal(40, upfps, angles)

        angvel = calc_ang_velocity(angles)
        angv = np.abs(line_smoother(angvel, window_size=11))
        angac = np.abs(np.diff(angv))

        try:
            ymove = np.argwhere(y > yth)[0][0]
            xmove = np.argwhere(np.abs(x) > xth)[0][0]
            xfast = np.argwhere(line_smoother(np.diff(x)) > 2)[0][0]

            thetafast = np.concatenate([find_peaks(angac, height=thetath, distance=5)[0], find_peaks(-angac, height=thetath, distance=5)[0]])
        except:
            continue

        delta_x_f.append(xmove); delta_y_f.append(ymove)
        col = ch.get_rgb(xmove)

        # plot
        axarr2[0].plot(np.arange(len(angac))-ymove, angac, color=col)
        axarr2[0].scatter(thetafast-ymove, angac[thetafast], color=red,  zorder=99)

        axarr2[1].plot(np.arange(len(angv))-ymove, angv, color=col)
        axarr2[1].scatter(thetafast-ymove, angv[thetafast], color=red,  zorder=99)

        polax.scatter(upangles[ymove:], np.arange(len(upangles[ymove:])), c=np.arange(len(upangles[ymove:])), cmap=cmaps[i])
        # polax.plot(upangles, np.arange(len(upangles)), color=grey)

        uptheta = [int(x/40*upfps) for x in thetafast]
        polax.scatter(upangles[uptheta], uptheta, color=red)

        axarr.main.plot(x, y, lw=2, color=col, alpha=.40, zorder=99)
        axarr.main.scatter(x[thetafast], y[thetafast], color=red, zorder=99)


        axarr.x.plot(np.abs(x), color=col, alpha=.40)
        axarr.y.plot(y, color=col, alpha=.40)
        break

axarr2[0].axvline(0, color=white)
axarr2[1].axvline(0, color=white)


axarr.x.axhline(xth, lw=2, ls="--", color=red)
axarr.y.axhline(yth, lw=2, ls="--", color=red)
axarr.main.axvline(xth, lw=2, ls="--", color=red)
axarr.main.axhline(yth, lw=2, ls="--", color=red)

axarr.main.set(title=condition, xlim=[-40, 40], ylim=[-50, 175], facecolor=[.2, .2, .2])
axarr.x.set(facecolor=[.2, .2, .2], ylim=[0, 40])
axarr.y.set(facecolor=[.2, .2, .2], ylim=[-50, 175])
axarr2[0].set(facecolor=[.2, .2, .2], xlim=[-25, 75], ylim=[0, 7.5])
axarr2[1].set(facecolor=[.2, .2, .2])
polax.set(facecolor=[.2, .2, .2], ylim=[0, ymove/40*upfps])




#%%
