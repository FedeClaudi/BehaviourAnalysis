import sys
sys.path.append('./')

from Utilities.imports import *

from Processing.trials_analysis.all_trials_loader import Trials
from Modelling.maze_solvers.gradient_agent import GradientAgent

class Torosity(Trials):
    def __init__(self):
        Trials.__init__(self, exp_1_mode=True)
        self.agent = GradientAgent(grid_size=1000, start_loc=[500, 800], goal_loc=[500, 262], stride=1)



    def plot_an_escape_on_the_options(self):
        f, axarr = plt.subplots(ncols=2)
        colors = get_n_colors(5)
        bridges = ("Left_Far", "Right_Medium")

        for i in range(1):
            for ax, br in zip(axarr, bridges):
                trials = self.trials.loc[(self.trials['grouped_experiment_name']=="asymmetric") & (self.trials['escape_arm']==br)]
                trial = trials.iloc[np.random.randint(0, len(trials))]


                # get the start and end of the escape
                start, end = list(trial.tracking_data[0, :2, 0].astype(np.int16)), list(trial.tracking_data[-1, :2, 0].astype(np.int16))


                # get the new geodistance to the location where the escape ends
                geod = self.agent.get_geo_to_point(end)
                if geod is None: continue


                # do a walk with the same geod
                walk = self.agent.walk(start = start, goal=end, geodesic_map = geod)

                _ = self.agent.plot_walk(walk, ax=ax)
                ax.scatter(trial.tracking_data[:, 0, 0], trial.tracking_data[:, 1, 0],  color=colors[i])
                a = 1
  


if __name__ == "__main__":
    t = Torosity()
    t.agent.plot_maze()

    t.plot_an_escape_on_the_options()

    plt.show()








