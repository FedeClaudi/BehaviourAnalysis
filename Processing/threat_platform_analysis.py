# %%
import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project
from Utilities.imports import *

from Processing.psychometric_analysis import PsychometricAnalyser

# %matplotlib inline

# %%
class ThreatPlatformsAnalyser(PsychometricAnalyser):
    def __init__(self):
        PsychometricAnalyser.__init__(self, naive=None, lights=1, escapes=True, escapes_dur=True)

        self.prep_trials_data()


    def prep_trials_data(self):
        """
            Get the trials for all conditions and get, the tracking specific to the threat, 
            the time at which the mouse leaves T, the speed at which the mouse leaves T
        """
        self.trials={}
        for condition, data in self.conditions.items():
            out_of_ts, threat_trackings, speeds_at_out_t = [], [], []

            trials = self.get_sessions_trials(maze_design=int(condition[-1]), naive=None, lights=1, escapes=True, escapes_dur=True)
            
            for i, trial in trials.iterrows():
                out_of_t = np.int(trial.time_out_of_t*trial.fps)
                tracking = trial.tracking_data[:out_of_t, :3]

                out_of_ts.append(out_of_t)
                threat_trackings.append(tracking)
                speeds_at_out_t.append(tracking[-1, -1])

            trials["frame_out_of_t"] = out_of_ts
            trials["threat_tracking"] = threat_trackings
            trials["speed_at_out_t"] = speeds_at_out_t

            self.trials[condition] = trials

        self.all_trials = pd.concat(list(self.trials.values()))

    def inspect_speed(self):
        f, axarr = create_figure(subplots=True, ncols=4)

        for condition, trials in self.trials.items():
            for i, trial in trials.iterrows():
                axarr[0].plot(trial.threat_tracking[:, 0], trial.threat_tracking[:, 1], color=black, lw=1, alpha=.1)
                axarr[3].scatter(trial.speed_at_out_t, np.nanmedian(trial.tracking_data[:, 2]), color=black, s=15, alpha=.8)

            axarr[1].scatter(trials.escape_duration, trials.time_out_of_t, color=black, s=15, alpha=.8)
            axarr[2].scatter(trials.speed_at_out_t, trials.time_out_of_t, color=black, s=15, alpha=.8)
 

        axarr[0].set(xlabel="X", ylabel="Y", xlim=[440, 560])
        axarr[1].set(xlabel="tot duration (s)", ylabel="out of T (s)", ylim=[0, 8])
        axarr[2].set(ylabel="out of T (s)", xlabel="speed at out of t", ylim=[0, 8])
        axarr[3].set(ylabel="speed at out of t", xlabel="mean escape speed", ylim=[0, 8])

        clean_axes(f)
        f.tight_layout()


    def torosity_analysis(self, time_end=None):
        # ! dont know what I;m doing
        f, axarr = create_figure(subplots=True, ncols=3)

        integral_angular_velocity = []
        for i, trial in self.all_trials.iterrows():
            if time_end is not None:  # take ang vel up to N second
                end = np.int(time_end * trial.fps)
                if end < 3: continue
            else:
                end = -1

            angles = calc_angle_between_points_of_vector(trial.threat_tracking[:end, :2])
            ang_vel = calc_ang_velocity(line_smoother(angles, window_size=11))
            integral_angular_velocity.append(np.sum(np.abs(ang_vel))/len(ang_vel))

            axarr[2].plot(trial.threat_tracking[:, 2], ang_vel, color=black, alpha=0.4)

      
        axarr[0].hist(integral_angular_velocity, color=black)

        axarr[1].scatter(integral_angular_velocity, self.all_trials.time_out_of_t, color=black, s=10, alpha=.6)

            



if __name__ == "__main__":
    c = ThreatPlatformsAnalyser()

# %%
if __name__ == "__main__":
    c.torosity_analysis(time_end=20)




#%%
