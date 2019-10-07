# %%
import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project
from Utilities.imports import *

from scipy.signal import medfilt as median_filter

from Processing.psychometric_analysis import PsychometricAnalyser
from Modelling.maze_solvers.environment import Environment

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
