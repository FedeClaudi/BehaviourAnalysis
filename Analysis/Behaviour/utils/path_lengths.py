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
			try:
				return percentile_range(path_lengths, low=10, high=90)
			except:
				return (0)
				
		res = namedtuple("res", "left right center ratio")
		res2 = namedtuple("percentile", "low median mean high std sem")
		self.get_tracking_after_leaving_T_for_conditions()

		results = {}
		for condition, trials in self.conditions.items():
			left_trials = trials.loc[trials.escape_arm == 'left']
			right_trials = trials.loc[trials.escape_arm == 'right']
			center_trials = trials.loc[trials.escape_arm == 'center']

			lres, rres, cres = do_arm(left_trials), do_arm(right_trials), do_arm(center_trials)

			ratio = res2(*[l/r for l,r in zip(lres, rres)])
			results[condition] = res(lres, rres, cres, ratio)
		return results

	def get_duration_per_arm_from_trials(self):
		"""[Estimates the average and confidence intervals of escape durations for each arm.
			It only looks at the interval between when the mice step off the threat platform
			to when they step onto the shelter platform.]
		"""

		def do_arm(data):
			durations = []
			for i, trial in data.iterrows():
				durations.append(trial.after_t_tracking.shape[0]/trial.fps)
			try:
				return percentile_range(durations, low=10, high=90)
			except:
				return (0)

		res = namedtuple("res", "left right center ratio")
		res2 = namedtuple("percentile", "low median mean high std sem")
		
		self.get_tracking_after_leaving_T_for_conditions()

		results = {}
		alldata = {c:{a:[] for a in ['left', 'right', 'center']} for c in self.conditions.keys()}
		for condition, trials in self.conditions.items():
			left_trials = trials.loc[trials.escape_arm == 'left']
			right_trials = trials.loc[trials.escape_arm == 'right']
			center_trials = trials.loc[trials.escape_arm == 'center']

			lres, rres, cres = do_arm(left_trials), do_arm(right_trials), do_arm(center_trials)

			ratio = res2(*[l/r for l,r in zip(lres, rres)])
			results[condition] = res(lres, rres, cres, ratio)

			for i, trial in trials.iterrows():
				alldata[condition][trial.escape_arm].append(trial.after_t_tracking.shape[0]/trial.fps)
		return results, alldata

	def get_exploration_per_path_from_trials(self):
		res = namedtuple("res", "left right ratio")
		results = {}
		for i, (condition, trials) in enumerate(self.conditions.items()):
			uids = set(trials.uid.values)
			exps = self.explorations[self.explorations['uid'].isin(uids)]
			print("Exp. {} - {} mice".format(condition, len(set(exps.mouse_id.values))))

			time_on_l, time_on_r, ratios = [], [], []
			for ii, exp in exps.iterrows():
				if exp.end_frame == -1: 
					print("skipping no end")
					continue
				tracking = (TrackingData * TrackingData.BodyPartData & "bpname='body'" & "uid='{}'".format(exp.uid)).fetch("tracking_data")
				tracking = np.vstack(tracking)[exp.start_frame:exp.end_frame, :]

				on_the_left = tracking[tracking[:, 0] < 450, :]
				on_the_right = tracking[tracking[:, 0] > 550, :]
				fps = trials.loc[trials.uid == exp.uid].fps.values[0]

				time_on_l.append(on_the_left.shape[0]/fps)
				time_on_r.append(on_the_right.shape[0]/fps)
				ratios.append(time_on_l[-1]/time_on_r[-1])

			left, right, ratio = percentile_range(time_on_l), percentile_range(time_on_r), percentile_range(ratios)
			results[condition] = res(left, right, ratio)
		return results
