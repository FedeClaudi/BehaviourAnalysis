import sys
sys.path.append('./')

from scipy.signal import resample

from Utilities.imports import *

from Processing.trials_analysis.all_trials_loader import Trials
from Modelling.maze_solvers.gradient_agent import GradientAgent

class Torosity(Trials):
    def __init__(self):
        Trials.__init__(self, exp_1_mode=True)

        # Create scaled agent
        self.scale_factor = 0.25
        self.agent = GradientAgent(grid_size =  int(1000*self.scale_factor), 
                                    start_loc= [int(500*self.scale_factor), int(800*self.scale_factor)], 
                                    goal_loc = [int(500*self.scale_factor), int(260*self.scale_factor)], stride=1)

        scaled_blocks = {}
        for k, states in self.agent.bridges_block_states.items():
            scaled_blocks[k] = [(int(x*self.scale_factor), int(y*self.scale_factor)) for x,y in states]
        self.agent.bridges_block_states = scaled_blocks


    def smallify_tracking(self, tracking):
        return np.multiply(tracking, self.scale_factor).astype(np.int32)

    def process_one_trial(self, trial, br):
        # scale down the tracking data
        tracking = self.smallify_tracking(trial.tracking_data.astype(np.int16))

        # get the start and end of the escape
        self.agent.start_location, self.agent.goal_location = list(tracking[0, :2, 0]), list(tracking[-1, :2, 0])

        # get the new geodistance to the location where the escape ends
        self.agent.geodesic_distance = self.agent.get_geo_to_point(self.agent.goal_location)
        if self.agent.geodesic_distance is None: return None, None

        # Introduce blocked bridge if LEFT escape
        if "left" in br.lower():
            self.agent.introduce_blockage("right_large", update =True)
            # self.agent.plot_maze()

        # do a walk with the same geod
        walk = self.agent.walk()

        # re sample the walk to match the n frames in th
        n_frames = tracking.shape[0]


        return tracking, walk


    def plot_one_escape_and_its_walk(self):
        f, axarr = plt.subplots(ncols=2)
        colors = get_n_colors(5)
        bridges = ("Left_Far", "Right_Medium")

        for i in range(1):
            for ax,  br in zip(axarr,  bridges):
                # Reset 
                self.agent._reset()

                # Get some tracking
                trials = self.trials.loc[(self.trials['grouped_experiment_name']=="asymmetric") & (self.trials['escape_arm']==br)]
                trial = trials.iloc[np.random.randint(0, len(trials))]

                # Process
                tracking, walk = self.process_one_trial(trial, br)
               
                if tracking is None: continue
                # Plot
                _ = self.agent.plot_walk(walk, ax=ax)
                ax.scatter(tracking[:, 0, 0], tracking[:, 1, 0],  c='r', s=100, alpha=.25)
                
  


if __name__ == "__main__":
    t = Torosity()

    t.plot_one_escape_and_its_walk()

    plt.show()









