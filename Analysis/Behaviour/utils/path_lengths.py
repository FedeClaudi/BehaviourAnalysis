import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project

from Utilities.imports import *

from Analysis.Behaviour.utils.behaviour_variables import *
from Modelling.maze_solvers.gradient_agent import GradientAgent


class PathLengthsEstimator:
	ratio = "georatio"  # ? Use  either georatio or ratio for estimating L/R length ratio
	def __init__(self):
		self.agent_path_lengths = None
		self.trials_paths_lengths = None

	def get_arms_lengths_with_agent(self, load=False):
		if not load:
			# Get Gradiend Agent and maze arms images
			agent = GradientAgent(grid_size=1000, start_location=[515, 208], goal_location=[515, 720])

			if sys.platform == "darwin":
				maze_arms_fld = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/analysis_metadata/maze_solvers/good_single_arms"
			else:
				maze_arms_fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\maze_solvers\\good_single_arms"
			
			arms = [os.path.join(maze_arms_fld, a) for a in os.listdir(maze_arms_fld) if "jpg" in a or "png" in a]
			arms_data = dict(maze=[], n_steps=[], torosity=[], distance=[])
			# Loop over each arm
			for arm in arms:
				if "centre" in arm: continue
				print("getting geo distance for arm: ",arm)
				# ? get maze, geodesic and walk
				agent.maze, agent.free_states = agent.get_maze_from_image(model_path=arm)
				agent.geodesic_distance = agent.geodist(agent.maze, agent.goal_location)
				walk = agent.walk()
				agent.plot_walk(walk)


				# Process walks to get lengths and so on
				arms_data["maze"].append(os.path.split(arm)[1].split(".")[0])
				arms_data["n_steps"].append(len(walk))
				arms_data["distance"].append(round(np.sum(calc_distance_between_points_in_a_vector_2d(np.array(walk)))))
				threat_shelter_dist = calc_distance_between_points_2d(agent.start_location, agent.goal_location)
				arms_data["torosity"].append(np.sum(calc_distance_between_points_in_a_vector_2d(np.array(walk))) / threat_shelter_dist)
			
			self.agent_path_lengths = pd.DataFrame(arms_data)
		else:
			self.agent_path_lengths =  pd.read_pickle(os.path.join(self.metadata_folder, "geoagent_paths.pkl"))

	def get_arms_lengths_from_trials(self):
		""" Estimates the path length for the left and right paths from the data. 
			It does so by only looking at the part of the escape trials between when the mice lave
			the threat platform to when they get to the shelter platform. It returns the data for left
			and right as 5th 95th percentile and media and the ratio of these values.
		"""
		def do_arm(data):
			path_lengths = []
			for i, trial in data.iterrows():
				path_lengths.append(np.sum(calc_distance_between_points_in_a_vector_2d(trial.after_t_tracking)))
			return percentile_range(path_lengths)

		res = namedtuple("res", "left right ratio")
		res2 = namedtuple("percentile", "low median mean high")
		self.get_tracking_after_leaving_T_for_conditions()

		results = {}
		for condition, trials in self.conditions.items():
			left_trials = trials.loc[trials.escape_arm == 'left']
			right_trials = trials.loc[trials.escape_arm == 'right']

			lres, rres = do_arm(left_trials), do_arm(right_trials)

			ratio = res2(*[l/r for l,r in zip(lres, rres)])
			results[condition] = res(lres, rres, ratio)
		return results


	def get_lengths_ratios(self):
		if self.agent_path_lengths is None:
			self.get_arms_lengths_with_agent()

		# TODO make this more general
		short_arm_dist = self.paths_lengths.loc[self.paths_lengths.maze=="maze4"].distance.values[0]
		self.agent_path_lengths["georatio"] = [round(x / short_arm_dist, 4) for x in self.paths_lengths.distance.values]
		
		self.short_arm_len = self.paths_lengths.loc[self.paths_lengths.maze=="maze4"][self.ratio].values[0]
		self.long_arm_len = self.paths_lengths.loc[self.paths_lengths.maze=="maze1"][self.ratio].values[0]

