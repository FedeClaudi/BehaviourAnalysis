import sys
sys.path.append('./')

from Utilities.imports import *


"""
	Loads all the trials from the table and organises them in a database that can be easily handled
	# ! Works with ALLTRIALS table
"""

class Trials:
	def __init__(self, selected_experiments=None, just_escapes=True, exp_mode=None):
		 
		self.selected_experiments = selected_experiments
		self.just_escapes = just_escapes

		# Some sessions have different experiment names but can be grouped
		self.grouped_experiments = dict(
			asymmetric = ["PathInt2", "PathInt2-L"],
			symmetric = ["Square Maze", "TwoAndahalf Maze"],
			threearms = ["PathInt"]
		)


		# ? facilitate the loading of asymmetric and symmetric data
		if exp_mode == 1:  # Two arms maze asyma and sym
			self.selected_experiments = self.grouped_experiments['asymmetric'].copy()
			self.selected_experiments.extend(self.grouped_experiments['symmetric'])
		elif exp_mode == 0:  # +  3 arms maze
			self.selected_experiments = self.grouped_experiments['asymmetric'].copy()
			self.selected_experiments.extend(self.grouped_experiments['symmetric'])
			self.selected_experiments.extend(self.grouped_experiments['threearms'])

		# Define which items to load from the table for each entry
		self.elems_to_load = ["trial_id", "tracking_data","session_uid", "experiment_name", "escape_arm", 
								"origin_arm", "fps", "is_escape", "time_out_of_t", "outward_tracking_data", "origin_arm"]

		# Load data
		self.trials = self.load()

	def load(self, selected_experiments = None, just_escapes=None):

		# Allow the user to select the experiments at loading time
		if selected_experiments is None: selected_experiments = self.selected_experiments
		if just_escapes is None: just_escapes = self.just_escapes

		print("Loading trial data for exp:", selected_experiments)

		# load as dataframe
		if selected_experiments is None: # load all
			trials =  pd.DataFrame((AllTrials & "is_escape='{}'".format(just_escapes) ).fetch(*self.elems_to_load)).T
		else:
			trials = []
			for exp in selected_experiments:
				if just_escapes != "all":
					temp_df = pd.DataFrame((AllTrials & "is_escape='{}'".format(just_escapes) & "experiment_name='{}'".format(exp) ).fetch(*self.elems_to_load)).T
				else:#
					temp_df = pd.DataFrame((AllTrials & "experiment_name='{}'".format(exp) ).fetch(*self.elems_to_load)).T

				temp_df.columns = self.elems_to_load
				trials.append(temp_df)

			# df_col_merged =pd.concat([df_a, df_b], axis=1)
			trials = pd.concat(trials, axis=0)

		# change the experiment name for those who need to be grouped
		grouped_exp_names = []
		for e in trials['experiment_name'].values:
			check = False
			for k, names in self.grouped_experiments.items():
				if e in names:
					grouped_exp_names.append(k)
					check = True
					break
			if not check: grouped_exp_names.append(e)
		trials['grouped_experiment_name'] = grouped_exp_names
		return trials

	def save_trials(self, save_path):
		saved_check = save_yaml(save_path, self.data)
		if not saved_check: print("Could not save data at this time sorry. Path: ". save_path)

	def plot_one(self):
		f, ax = plt.subplots()
		trial = self.trials.iloc[np.random.randint(0, len(self.trials))]
		ax.scatter(trial.tracking_data[:, 0], trial.tracking_data[:,  1])

	def plot_parm_experiment(self):
		experiments = set(self.trials.grouped_experiment_name.values)

		f,axarr = plt.subplots(figsize=(10, 8), ncols=len(experiments), sharey=True)

		colors = {"Left_Medium":"g", "Left_Far":"r", "Centre":"m", "Right_Medium":"g"}
		for exp,ax in zip(experiments, axarr):
			outcomes = self.trials.loc[self.trials.grouped_experiment_name == exp].escape_arm.values
			arms = sorted(set(outcomes))
			p_arm = [calc_prob_item_in_list(outcomes, arm) for arm in arms]
			x = np.arange(len(arms))
			cc = [colors[arm] for arm in arms]
			
			ax.bar(x, p_arm, color=cc)
			ax.set(xticks=x, xticklabels=arms, title=exp + " - n: {}".format(len(outcomes)), facecolor=[.2, .2, .2])
			ax.tick_params(axis='x', rotation=45)



if __name__ == "__main__":
	t = Trials(exp_1_mode=True)
	print(t.trials)
