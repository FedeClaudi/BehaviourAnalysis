# %%
import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project
from Utilities.imports import *

from scipy.signal import medfilt as median_filter

from Processing.psychometric_analysis import PsychometricAnalyser
from Modelling.maze_solvers.environment import Environment

%matplotlib inline



# %%
class ThreatPlatformsAnalyser(PsychometricAnalyser, Environment):
    def __init__(self):
        PsychometricAnalyser.__init__(self, naive=None, lights=1, escapes=True, escapes_dur=None)
        Environment.__init__(self, grid_size=1000, maze_design="PathInt2_old.png")

        self.maze = np.rot90(self.maze, 2)

        self.prep_trials_data()


    def prep_trials_data(self):
        """
            Get the trials for all conditions and get, the tracking specific to the threat, 
            the time at which the mouse leaves T, the speed at which the mouse leaves T
        """
        self.trials={}
        for condition, data in self.conditions.items():
            out_of_ts, threat_trackings, s_threat_trackings, t_threat_trackings, speeds_at_out_t = [], [], [],  [], []

            trials = self.get_sessions_trials(maze_design=int(condition[-1]), naive=None, lights=1, escapes=True, escapes_dur=True)
            
            for i, trial in trials.iterrows():
                out_of_t = np.int(trial.time_out_of_t*trial.fps)
                tracking = trial.tracking_data[:out_of_t, :3]
                s_tracking = trial.snout_tracking_data[:out_of_t, :3]
                t_tracking = trial.tail_tracking_data[:out_of_t, :3]

                out_of_ts.append(out_of_t)
                threat_trackings.append(tracking)
                s_threat_trackings.append(s_tracking); t_threat_trackings.append(t_tracking)
                speeds_at_out_t.append(tracking[-1, -1])

            trials["frame_out_of_t"] = out_of_ts
            trials["threat_tracking"] = threat_trackings
            trials["snout_threat_tracking"], trials["tail_threat_tracking"] = s_threat_trackings, t_threat_trackings
            trials["speed_at_out_t"] = speeds_at_out_t

            self.trials[condition] = trials

        self.all_trials = pd.concat(list(self.trials.values()))

    def inspect_speed(self):
        f, axarr = create_figure(subplots=True, nrows=2, figsize=(16, 16))
        # Plot left and right escape trajectories
        for condition, trials in c.trials.items():
            for i, trial in trials.iterrows():
                if "Left" in trial.escape_arm: 
                    color = red
                    xshift = -100
                    alpha=.3
                else:
                    color=magenta
                    xshift = 100
                    alpha=.15

                x,y = trial.threat_tracking[:, 0].copy(), trial.threat_tracking[:, 1].copy()

                # make all data in 40fps
                if trial.fps ==30:
                    x, y = upsample_signal(trial.fps, 40, x), upsample_signal(trial.fps, 40, y)
                elif trial.fps < 30: continue

                y[x > 560] = np.nan
                x[x > 560] = np.nan

                if np.all(np.isnan(x)): continue

                axarr[0].plot(x+xshift, y, color=color, lw=2, alpha=alpha)

                # Plot distance from end goal over time
                exitpoint = np.array([x[-1], y[-1]])
                d = calc_distance_from_shelter(np.array([x, y]).T, exitpoint)
                normed_distance = d/d[0]
                if np.all(np.isnan(normed_distance)): continue

                if trial.fps < 40: print("cacca")


                under_th = np.where(normed_distance < 0.75)[0][0]
                _x = np.arange(len(normed_distance))


                axarr[1].plot(_x-under_th, normed_distance, color=color, alpha=.1, lw=2)
                

        axarr[0].set(xlabel="X", ylabel="Y", aspect="equal", xlim=[250, 750],  ylim=[100, 450], facecolor=[.1, .1, .1])
        axarr[1].set(xlabel="time", ylabel="Distance from arm",  facecolor=[.1, .1, .1], xlim=[-75, 75])

        clean_axes(f)
        f.tight_layout()



            



if __name__ == "__main__":
    c = ThreatPlatformsAnalyser()


#%%
f, axarr = create_figure(subplots=True, ncols=2, sharex=True, sharey=True, figsize=(16, 16))
f2, ax2 = create_figure(subplots=False, figsize=(16, 16))


# Plot left and right escape trajectories
for i, trial in c.all_trials.iterrows():
    if "Left" in trial.escape_arm: 
        color = red
        xshift =  0
        alpha=.3
    else:
        color=magenta
        xshift = 0
        alpha=.15

    x,y = trial.threat_tracking[:, 0].copy(), trial.threat_tracking[:, 1].copy()
    sx, sy, tx, ty =  trial.snout_threat_tracking[:, 0].copy(), trial.snout_threat_tracking[:, 1].copy(), trial.tail_threat_tracking[:, 0].copy(), trial.tail_threat_tracking[:, 1].copy()


    if y[0] > 240: continue

    
    to_elim = np.where((x > 520)&(y<250))
    y[to_elim] = np.nan
    x[to_elim] = np.nan
    sx[to_elim], sy[to_elim], tx[to_elim], ty[to_elim] = np.nan, np.nan, np.nan, np.nan

    if np.all(np.isnan(x)): continue
    if not "Left" in trial.escape_arm and np.nanmin(x) < 460: continue

    axarr[0].plot(x+xshift, y, color=color, lw=2, alpha=alpha)
    # ax2.plot(x+xshift, y, color=color, lw=0.5, alpha=.8)


    # Get ang vel
    angles = calc_angle_between_vectors_of_points_2d(np.array([x, y]), np.array([tx, ty]))
    ang_speed = np.abs(calc_ang_velocity(angles))
    # ax2.scatter(x+xshift, y, c=angles+180, cmap="coolwarm", s=5, vmin=0, vmax=180,  alpha=alpha)
    ax2.plot(angles)



        

axarr[0].set(xlabel="X", ylabel="Y", aspect="equal", xlim=[250, 750],  ylim=[100, 450], facecolor=[.1, .1, .1])
axarr[1].set(xlabel="X", ylabel="Y", aspect="equal", xlim=[250, 750],  ylim=[100, 450], facecolor=[.1, .1, .1])

ax2.set(facecolor=[.1, .1, .1])

clean_axes(f)
f.tight_layout()

#%%
a, b = np.zeros((10, 2)), np.zeros((10, 2))
b[:, 1] = 1
calc_angle_between_vectors_of_points_2d(a.T, b.T)

#%%
