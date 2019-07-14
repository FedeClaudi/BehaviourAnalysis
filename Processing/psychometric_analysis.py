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
        f, axarr = plt.subplots(ncols=4)

        # Loop over each maze design and get the path lenght (from the tracking data)
        summary = dict(maze=[], left=[], right=[], ratio=[])
        for i in np.arange(4):
            mazen = i +1
            trials = self.get_sesions_trials(maze_design=mazen, naive=None, lights=None, escapes=True)

            # Get the length of each escape
            lengths = []
            for _, trial in trials.iterrows():
                lengths.append(np.sum(calc_distance_between_points_in_a_vector_2d(trial.tracking_data[:, :2])))
            trials["lengths"] = lengths

            left_trials, right_trials = [t.trial_id for i,t in trials.iterrows() if "left" in t.escape_arm.lower()], [t.trial_id for i,t in trials.iterrows() if "right" in t.escape_arm.lower()]
            left, right = trials.loc[trials.trial_id.isin(left_trials)], trials.loc[trials.trial_id.isin(right_trials)]

            # Plot stuff
            axarr[i].hist(left.lengths, density=True, color="g", alpha=.8)
            axarr[i].hist(right.lengths, density=True, color="r", alpha=.8)
            axarr[i].set(title=self.maze_designs[mazen])

            # Make dict for summary df
            l, r  = percentile_range(left.lengths, low=75).low, percentile_range(right.lengths, low=75).low
            summary["maze"].append(self.maze_designs[mazen])
            summary["left"].append(round(l, 2))
            summary["right"].append(round(r, 2))
            summary["ratio"].append(round(l / r, 4))

        self.paths_lengths = pd.DataFrame.from_dict(summary)


if __name__ == "__main__":
    pa = PsychometricAnalyser()
    pa.get_paths_lengths()
    print(pa.paths_lengths)
    plt.show()

