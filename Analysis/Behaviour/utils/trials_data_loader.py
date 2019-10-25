import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project

from Utilities.imports import *

from Analysis.Behaviour.utils.behaviour_variables import *

class TrialsLoader:
	max_duration_th = 19 # ? only trials in which the mice reach the shelter within this number of seconds are considered escapes (if using escapes == True)

	# look up dictionaries
	naive_lookup = {0: "experienced", 1:"naive", -1:"nan"}
	lights_lookup = {0: "off", 1:"on", 2:"on_trials", 3:"on_exploration", -1:"nan"}

	def __init__(self, load_psychometric=False, **kwargs):
		self.maze_designs = maze_designs

		if load_psychometric:
			self.conditions = dict(
								maze1 =  self.load_trials_by_condition(maze_design=1, **kwargs),
								maze2 =  self.load_trials_by_condition(maze_design=2, **kwargs),
								maze3 =  self.load_trials_by_condition(maze_design=3, **kwargs),
								maze4 =  self.load_trials_by_condition(maze_design=4, **kwargs),
							)

	def load_trials_by_condition(self, maze_design=None, naive=None, lights=None, escapes=None, escapes_dur=None, shelter=None):
		"""[Given a number of criteria, load the trials that match these criteria]
		
		Keyword Arguments:
			maze_design {[int]} -- [Number of maze of experiment] (default: {None})
			naive {[int]} -- [1 for naive only mice] (default: {None})
			lights {[int]} -- [1 for light on only experiments] (default: {None})
			escapes {bool} -- [if true only escape trials are used] (default: {False})
			escapes_dur {bool} -- [If true only trials in which the escapes terminate within the duraiton th are used] (default: {True})
		"""

		if naive is None: naive= self.naive
		if lights is None: lights= self.lights
		if escapes_dur is None: escapes_dur = self.escapes_dur
		if escapes is None: escapes = self.escapes
		if shelter is None: shelter = self.shelter

		# Get all trials from the AllTrials Table
		all_trials = pd.DataFrame(AllTrials.fetch())
		if escapes:
			all_trials = all_trials.loc[all_trials.is_escape == "true"]
		if escapes_dur:
			all_trials = all_trials.loc[all_trials.escape_duration <= self.max_duration_th]

		# Get the sessions that match the criteria and use them to discard other trials
		sessions = self.get_sessions_by_condition(maze_design=maze_design, naive=naive, lights=lights,shelter=shelter, df=True)
		ss = set(sorted(sessions.uid.values))
		trials = all_trials.loc[all_trials.session_uid.isin(ss)]
		return trials

	def get_sessions_by_condition(self, maze_design=None, naive=None, lights=None,  shelter=None, df=False):
		""" Query the DJ database table AllTrials for the trials that match the conditions """
		data = Session * Session.Metadata * Session.Shelter  - 'experiment_name="Foraging"'  - "maze_type=-1"

		if maze_design is not None:
			data = (data & "maze_type={}".format(maze_design))

		if naive is not None:
			data = (data & "naive={}".format(naive))

		if lights is not None:
			data = (data & "lights={}".format(lights))

		if shelter is not None:
			if shelter:
				data = (data & "shelter={}".format(1))
			else:
				data = (data & "shelter={}".format(0))
				
		if not len(data): print("Query didn't yield any results!")

		if df:
			return pd.DataFrame((data).fetch())
		else: 
			return data

	# ? Load trials data previously pickled (and corresponding save function)
	def load_trials_from_pickle(self, load_psychometric=False, load_path=None):
		if load_psychometric:
			names = ["maze1", "maze2", "maze3", "maze4"]
			return {n:load_df(os.path.join(self.metadata_folder, n+".pkl")) for n in names}	
		elif load_path is not None:
			return load_df(load_path)

	def save_trials_to_pickle(self, save_psychometric=False, save_path=None):
		if save_psychometric:
			for k, df in self.conditions.items():
				save_df(df, os.path.join(self.metadata_folder, k+".pkl"))
		elif save_path is not None:
			save_df(save_path)