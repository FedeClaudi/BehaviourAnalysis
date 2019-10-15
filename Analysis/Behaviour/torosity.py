import sys
sys.path.append('./')

from scipy.signal import resample

from Utilities.imports import *

from Processing.trials_analysis.all_trials_loader import Trials
from Modelling.maze_solvers.gradient_agent import GradientAgent

print("\n\n\n")

class Torosity(Trials):
	"""[Utils to facilitate the computation of torosity for tracking data in a maze]
	"""

	def __init__(self, mtype = None):
		# Create scaled agent
		self.scale_factor = 0.25

		if mtype == "asymmetric" or mtype is None:
			self.agent = GradientAgent(
										maze_type = "asymmetric_large",
										maze_design = "PathInt2.png",
										grid_size = int(1000*self.scale_factor), 
										start_loc= [int(500*self.scale_factor), int(700*self.scale_factor)], 
										goal_loc = [int(500*self.scale_factor), int(260*self.scale_factor)], stride=1)
		else: 
			raise NotImplementedError

		scaled_blocks = {}
		for k, states in self.agent.bridges_block_states.items():
			scaled_blocks[k] = [(int(x*self.scale_factor), int(y*self.scale_factor)) for x,y in states]
		self.agent.bridges_block_states = scaled_blocks

		if mtype == "asymmetric":
			self.bridges_lookup = dict(Right_Medium="right", Left_Far="left")
		elif mtype == "symmetric":
			self.bridges_lookup = dict(Right_Medium="right", Left_Medium="left")
		else:
			self.bridges_lookup = dict(Right_Medium="right", Left_Medium="left", Left_Far="left", Right_Far="right", Central="centre", Centre="centre")
		
		self.mtype = mtype

	"""
		#########################################################################################################################################################
				UTILS
		#########################################################################################################################################################
	"""

	def smallify_tracking(self, tracking):
		return np.multiply(tracking, self.scale_factor) # .astype(np.int32)

	@staticmethod
	def zscore_and_sort(res):
		res['walk_distance_z'] = stats.zscore(res.walk_distance)
		res['tracking_distance_z'] = stats.zscore(res.tracking_distance)
		res['torosity_z'] = stats.zscore(res.torosity)

		res.sort_values("torosity_z")

		return res

	"""
		#########################################################################################################################################################
				PROCESSING
		#########################################################################################################################################################
	"""


	def time_binned_torosity(self, tracking, br):
		window_size, window_step = 10, 5

		i_start = 0
		i_end = i_start + window_size

		binned_tor_list = []
		while i_end <= tracking.shape[0]:
			binned_tracking = tracking[i_start:i_end, :, :]
			i_start += window_step
			i_end += window_step

			torosity = self.process_one_trial(None, br, tracking=binned_tracking, goal=list(binned_tracking[-1, :2, 0]))

			binned_tor_list.append((i_start, i_end, torosity))

		return binned_tor_list

	def process_one_trial(self, br=None, trial=None, tracking=None, goal=None):
		# Reset 
		self.agent._reset()

		if tracking is None:
			# scale down the tracking data
			tracking = self.smallify_tracking(trial.tracking_data.astype(np.int16))
			outward_tracking = self.smallify_tracking(trial.outward_tracking_data.astype(np.int16))
		else:
			if len(tracking.shape) == 2:
				tracking = tracking[:, :, np.newaxis]

		# get the start and end of the escape
		self.agent.start_location = list(tracking[0, :2, 0])
		if goal is not None:
			self.agent.goal_location = goal
		else:
			self.agent.goal_location = list(tracking[-1, :2, 0])

		# get the new geodistance to the location where the escape ends
		self.agent.geodesic_distance = self.agent.get_geo_to_point(self.agent.goal_location)

		if self.agent.geodesic_distance is None and goal is None: return None, None, None
		elif self.agent.geodesic_distance is None and goal is not None: return np.nan

		# Introduce blocked bridge if LEFT escape
		if br is not None:
			if "left" in br.lower():
				self.agent.introduce_blockage("right_large", update=True)

		# do a walk with the same geod
		walk = np.array(self.agent.walk())

		# compute stuff
		walk_distance  = np.sum(calc_distance_between_points_in_a_vector_2d(walk)) 
		tracking_distance   = np.sum(calc_distance_between_points_in_a_vector_2d(tracking[:, :2, 0])) 
		torosity = tracking_distance/walk_distance

		if trial is not None:
			# Create results
			results = dict(
				walk_distance       = walk_distance,
				tracking_distance   = tracking_distance,
				torosity = torosity,
				tracking_data = tracking,
				outward_tracking = outward_tracking, 
				escape_arm = br,
				is_escape = trial.is_escape
			)

			return tracking, walk, results
		else:
			return torosity
