# %%
import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project

from Utilities.imports import *

from Processing.analyse_experiments import ExperimentsAnalyser

# %%
# Define class

class PsychometricAnalyser(ExperimentsAnalyser):
    def __init__(self):
        ExperimentsAnalyser.__init__(self)

    def get_paths_lengths(self):
        # Loop over each maze design and get the path lenght (from the tracking data)
        summary = dict(maze=[], left=[], right=[], ratio=[])
        for i in np.arange(4):
            mazen = i +1
            trials = self.get_sesions_trials(maze_design=mazen, naive=None, lights=None, escapes=True)

            # Get the length of each escape
            lengths = []
            for i, trial in trials.iterrows():
                lengths.append(np.sum(calc_distance_between_points_in_a_vector_2d(trial.tracking_data[:, :2])))
            trials["lengths"] = lengths

            meanl = np.median([t.lengths for i, t in trials.iterrows() if "left" in t.escape_arm.lower()])
            meanr = np.median([t.lengths for i, t in trials.iterrows() if "right" in t.escape_arm.lower()])
            summary["maze"].append(self.maze_designs[mazen])
            summary["left"].append(round(meanl, 2))
            summary["right"].append(round(meanr, 2))
            summary["ratio"].append(round(meanl / meanr, 4))

        self.paths_lengths = pd.DataFrame.from_dict(summary)


if __name__ == "__main__":
    pa = PsychometricAnalyser()
    pa.get_paths_lengths()
    print(pa.paths_lengths)

