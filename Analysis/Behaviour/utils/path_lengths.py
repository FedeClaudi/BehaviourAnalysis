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

	def get_arms_lengths_from_trials(self, load=False):
		if not load:
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

				# Make dict for summary df
				l, r  = percentile_range(left.lengths, low=10).low, percentile_range(right.lengths, low=10).low
				summary["maze"].append(self.maze_names_r[self.maze_designs[mazen]])
				summary["left"].append(round(l, 2))
				summary["right"].append(round(r, 2))
				summary["ratio"].append(round(l / r, 4))

			self.trials_paths_lengths = pd.DataFrame.from_dict(summary)
		else:
			self.trials_paths_lengths = pd.read_pickle(os.path.join(self.metadata_folder, "path_lengths.pkl"))

	def get_lengths_ratios(self):
		if self.agent_path_lengths is None:
			self.get_arms_lengths_with_agent()

		# TODO make this more general
		short_arm_dist = self.paths_lengths.loc[self.paths_lengths.maze=="maze4"].distance.values[0]
		self.agent_path_lengths["georatio"] = [round(x / short_arm_dist, 4) for x in self.paths_lengths.distance.values]
		
		self.short_arm_len = self.paths_lengths.loc[self.paths_lengths.maze=="maze4"][self.ratio].values[0]
		self.long_arm_len = self.paths_lengths.loc[self.paths_lengths.maze=="maze1"][self.ratio].values[0]



	def get_uclidean_distance_along_path_from_data(self):
		a = 1