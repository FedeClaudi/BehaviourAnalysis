import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project
from Utilities.imports import *
from Processing.plot.plot_distributions import plot_fitted_curve, dist_plot

class TimeSeriesAnalysis:
    def __init__(self):
        # ? Set up for torosity analysis
		# Create scaled agent
		self.scale_factor = 0.25

		# lookup vars
		self.results_keys = ["walk_distance", "tracking_distance", "torosity", "tracking_data", "escape_arm", "is_escape", "binned_torosity",
							"time_out_of_t", "threat_torosity", "outward_tracking", "origin_torosity"]

    def setup_torosity_analysis(self, maze_image):
        self.agent = GradientAgent(
                            maze_type = "asymmetric_large",
                            maze_design = maze_image,
                            grid_size = int(1000*self.scale_factor), 
                            start_loc= [int(500*self.scale_factor), int(700*self.scale_factor)], 
                            goal_loc = [int(500*self.scale_factor), int(260*self.scale_factor)], stride=1)

	def smallify_tracking(self, tracking):
		return np.multiply(tracking, self.scale_factor).astype(np.int32)

    def process_one_trial(self, tracking, goal=None):
            # Reset 
            self.agent._reset()

            # get the start and end of the escape
            self.agent.start_location = list(tracking[0, :2])

            if goal is not None:
                self.agent.goal_location = goal
            else:
                self.agent.goal_location = list(tracking[-1, :2])

            # get the new geodistance to the location where the escape ends
            self.agent.geodesic_distance = self.agent.get_geo_to_point(self.agent.goal_location)

            if self.agent.geodesic_distance is None: return None

            # do a walk with the new geod
            walk = np.array(self.agent.walk())

            # compute stuff
            walk_distance       = np.sum(calc_distance_between_points_in_a_vector_2d(walk)) 
            tracking_distance   = np.sum(calc_distance_between_points_in_a_vector_2d(tracking[:, :2, 0])) 
            torosity            = tracking_distance/walk_distance

            return torosity


    # TODO method that loops over each trial in the database, loads the correct image depending on the arm the mouse escaped on
    # then computes the torosity and stores all torosities. 