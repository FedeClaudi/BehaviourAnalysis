import sys
sys.path.append('./')

from scipy.signal import resample

from Utilities.imports import *

from Processing.trials_analysis.all_trials_loader import Trials
from Modelling.maze_solvers.gradient_agent import GradientAgent

print("\n\n\n")

class Torosity(Trials):
	def __init__(self, mtype, just_esc):
		if sys.platform != "darwin": 
			if mtype != "all":
				Trials.__init__(self, exp_1_mode=True, just_escapes=just_esc)
				self.trials = self.trials.loc[self.trials['grouped_experiment_name']==mtype]  # only keep the trials from asym exp

			else:
				good_experiments = [  'FlipFlop Maze', 'FlipFlop2 Maze', 'PathInt', 'PathInt2', 'Square Maze', 'TwoAndahalf Maze', "PathInt2 L", 'PathInt2-L', ]
				Trials.__init__(self, exp_1_mode=False, just_escapes=just_esc, selected_experiments=good_experiments)


		# Create scaled agent
		self.scale_factor = 0.25

		if mtype == "asymmetric" or mtype=="all":
			self.agent = GradientAgent(
										maze_type = "asymmetric_large",
										maze_design = "PathInt2.png",
										grid_size = int(1000*self.scale_factor), 
										start_loc= [int(500*self.scale_factor), int(800*self.scale_factor)], 
										goal_loc = [int(500*self.scale_factor), int(260*self.scale_factor)], stride=1)
		else:
			self.agent = GradientAgent(
										maze_type = "asymmetric_large",
										maze_design = "Square Maze.png",
										grid_size = int(1000*self.scale_factor), 
										start_loc= [int(500*self.scale_factor), int(730*self.scale_factor)], 
										goal_loc = [int(500*self.scale_factor), int(260*self.scale_factor)], stride=1)

		scaled_blocks = {}
		for k, states in self.agent.bridges_block_states.items():
			scaled_blocks[k] = [(int(x*self.scale_factor), int(y*self.scale_factor)) for x,y in states]
		self.agent.bridges_block_states = scaled_blocks

		# lookup vars
		self.results_keys = ["walk_distance", "tracking_distance", "torosity", "tracking_data", "escape_arm", "is_escape", "binned_torosity",
							"time_out_of_t", "threat_torosity", "outward_tracking", "origin_torosity"]

		if mtype == "asymmetric":
			self.bridges_lookup = dict(Right_Medium="right", Left_Far="left")
		elif mtype == "symmetric":
			self.bridges_lookup = dict(Right_Medium="right", Left_Medium="left")
		else:
			self.bridges_lookup = dict(Right_Medium="right", Left_Medium="left", Left_Far="left", Right_Far="right", Central="centre", Centre="centre")
		

		self.plots_save_fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\plots\\torosity\\check"
		self.results_fld = "Processing/trials_analysis/"
		self.results_name = "torosity"
		self.mtype = mtype

	"""
		#########################################################################################################################################################
				UTILS
		#########################################################################################################################################################
	"""

	def smallify_tracking(self, tracking):
		return np.multiply(tracking, self.scale_factor).astype(np.int32)

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

	def threat_torosity(self, tracking, out_of_t, br):
		tracking = tracking[:out_of_t, :, :]
		return self.process_one_trial(None, br, tracking=tracking, goal=list(tracking[-1, :2, 0]))

	def origin_torosity(self, outward_tracking, br):
		return self.process_one_trial(None, br, tracking=outward_tracking, goal=list(outward_tracking[-1, :2]))

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

	def process_one_trial(self, trial, br, tracking=None, goal=None):
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

	def analyse_all(self, plot=False, save_plots=False, bined_tor=False, threat_tor=False,):
		print("processing all trials")
		all_res = {k:[] for k in self.results_keys}

		trials = [t for i,t in self.trials.iterrows()]
		for trial in tqdm(trials):
			# Get the escape bridge
			try:
				bridge = self.bridges_lookup[trial.escape_arm]
				origin_bridge = self.bridges_lookup[trial.origin_arm.values[0]]
				time_out_of_t = int(trial.time_out_of_t * trial.fps)
			except: 
				print("		could not get bridge and time out of T")
				continue

			# Process whole trial
			tracking, walk, res = self.process_one_trial(trial, bridge)
			if tracking is None: 
				print("		something went wrong with trial processing")
				continue
			
			# Binned torosity
			if bined_tor:
				res["binned_torosity"] = self.time_binned_torosity(tracking, bridge)
				# Plot binned tor
				if plot:
					self.plot_time_binned_torosity(tracking, walk, res["binned_torosity"], res, save_plots=save_plots, trial_id=trial.trial_id)
			else:
				res["binned_torosity"] = None

			# Threat torosity
			if threat_tor:
				res['threat_torosity'] = self.threat_torosity(tracking, time_out_of_t, bridge)
			else:
				res['threat_torosity'] = None
			res['time_out_of_t'] = time_out_of_t

			# origin torosity
			res['origin_torosity'] = self.origin_torosity(res['outward_tracking'], origin_bridge)
			
			# Put all results into main dict
			for k,v in res.items(): all_res[k].append(v)
		results = pd.DataFrame.from_dict(all_res)

		# clean_up and save
		results = results.loc[results.torosity != np.inf]
		results.to_pickle(os.path.join(self.results_fld, self.results_name + "_{}.pkl".format(self.mtype)))

	"""
		#########################################################################################################################################################
				PLOTTING
		#########################################################################################################################################################
	"""
	def plot_one_escape_and_its_walk(self):  # one per arm
		f, axarr = plt.subplots(ncols=2)
		colors = get_n_colors(5)

		for i in range(1):
			for ax,  br in zip(axarr,  self.bridges_lookup.keys()):
				# Get some tracking
				trials = self.trials.loc[self.trials['escape_arm']==br]
				trial = trials.iloc[np.random.randint(0, len(trials))]

				# Process
				tracking, walk, res = self.process_one_trial(trial, br)
			   
				if tracking is None: continue
				# Plot
				_ = self.agent.plot_walk(walk, ax=ax)
				ax.scatter(tracking[:, 0, 0], tracking[:, 1, 0],  c='r', s=100, alpha=.25)

				ax.set(
					title = "Walk: {} - Track: {} - Tor: {}".format(round(res['walk_distance'], 2), round(res['tracking_distance'], 2), round(res['torosity'], 2))
				)
				
	def plot_time_binned_torosity(self, tracking, walk, binned_tor, res, save_plots=False, trial_id=None):
		f, ax = plt.subplots(figsize=(16, 16))

		rearranged = np.zeros((len(binned_tor), 3))

		for i, (i_start, i_end, tor) in enumerate(binned_tor):
			rearranged[i, :2] = tracking[i_start, :2, 0]
			try:
				rearranged[i, -1] = tor
			except:
				raise ValueError(tor)

		self.agent.plot_walk(walk, alpha=.6,  ax=ax)
		ax.scatter(rearranged[:, 0], rearranged[:, 1], c=rearranged[:, 2],  s=100, alpha=.8, cmap="inferno")
		ax.set(
			title = "Walk: {} - Track: {} - Tor: {}".format(round(res['walk_distance'], 2), round(res['tracking_distance'], 2), round(res['torosity'], 2))
		)
		
		if not save_plots:
			plt.show()
		else:
			f.savefig(os.path.join(self.plots_save_fld, '{}.png'.format(trial_id)))
			plt.close(f)



	"""
		#########################################################################################################################################################
				ANALYSIS
		#########################################################################################################################################################
	"""

	def results_loader(self, name, select_bridge=None, select_escapes=None):
		res = pd.read_pickle(os.path.join(self.results_fld, self.results_name + "_{}.pkl".format(name)))
		res['experiment'] = [name for i in range(len(res))]

		if select_bridge is not None:
			res = res.loc[res.escape_arm == select_bridge]

		if select_escapes is not None:
			if select_escapes:
				res = res.loc[res.is_escape == "true"]
			else:
				res = res.loc[res.is_escape == "false"]

		return res

	def load_all_res(self):
		res1 = self.zscore_and_sort(self.results_loader("asymmetric", select_bridge=None, select_escapes=True))
		res2 = self.zscore_and_sort(self.results_loader("symmetric", select_bridge=None, select_escapes=True))
		res = self.zscore_and_sort(pd.concat([res1, res2], axis=0))       

		ares1 = self.zscore_and_sort(self.results_loader("asymmetric", select_bridge=None, select_escapes=None))
		ares2 = self.zscore_and_sort(self.results_loader("symmetric", select_bridge=None, select_escapes=None))
		ares = self.zscore_and_sort(pd.concat([ares1, ares2], axis=0))    

		# res = self.zscore_and_sort(self.results_loader("all", select_bridge=None, select_escapes=True))
		# ares = self.zscore_and_sort(self.results_loader("all", select_bridge=None, select_escapes=None))

		return res, ares # ! , res1, ares1, res2, ares2

	def inspect_results(self):
		use_tor = "threat_torosity"

		# Get data
		res, ares, = self.load_all_res()
		tot_res = [x for x in res[use_tor] if not np.isnan(x)],
		esc_res =  [x for x in res.loc[res.is_escape == "true"][use_tor] if not np.isnan(x)]
		atot_res = [x for x in ares[use_tor] if not np.isnan(x)]

		print("\nTot trials: ", len(ares))

		# Focus on Torosity
		if use_tor == "threat_torosity":
			threshold = [(0.9, 1.0), (1.4, 1.5), (1.8, 2),  (3.5, 10)]
		else:
			threshold = [(-1.5, -0.7), (-.01, .01), (1, 1.1),  (4, 10)]
		colors = ['g', 'b', 'm', 'k']
		colormaps = ["Greens", "Blues", "Purples", "Greys"]

		# Create figure
		f, axarr = plt.subplots(ncols=4, nrows=2)
		axarr = axarr.flatten()

		# Plot escapes torosity
		_, bins, _ = axarr[0].hist(atot_res, bins=45, color='k', alpha=.55, log=True, label='all')
		_, bins, _ = axarr[0].hist(tot_res, bins=bins, color='yellow', alpha=.55, log=True, label='escapes')

		# Correlation between overall torosity and threat torosity
		sns.regplot(res.torosity.values, res.threat_torosity.values, ax=axarr[2], robust=True, n_boot=10, ci=None, truncate=True,
						line_kws={"color":"red", "linewidth":2,  "label":"Robust lin. reg."}, 
						scatter_kws={"color":"green", "alpha":.4, "s":150})
		sns.regplot(res.torosity.values, res.threat_torosity.values, ax=axarr[2], robust=False, n_boot=100, truncate=True, scatter=False, ci=None, 
				line_kws={"color":"blue", "linewidth":2, "label":"lin. reg."}, )

		# ? gives same results as regplot
		# x, intercept, slope = linear_regression(res.torosity.values, res.threat_torosity.values)
		# axarr[2].plot(x, intercept + slope*x, color="blue")

		# plot mean vel vs torosity
		mean_speeds = [np.mean(t.tracking_data[:, 2, 0]) for i,t in res.iterrows()]
		# axarr[3].scatter(mean_speeds, res.threat_torosity, color='k', s=150, alpha=.6)
		sns.regplot(mean_speeds, res.threat_torosity.values, ax=axarr[3], robust=True, n_boot=100, truncate=True, ci=None, 
				line_kws={"color":"red", "linewidth":2,  "label":"Robust lin. reg."}, 
				scatter_kws={"color":"black", "alpha":.4, "s":150})
		sns.regplot(mean_speeds, res.threat_torosity.values, ax=axarr[3], robust=False, n_boot=100, truncate=True, scatter=False, ci=None,
				line_kws={"color":"blue", "linewidth":2,  "label":"lin. reg."}, )

		# Plot ORIGIN vs ESCAPe torosity
		# ori_threat = remove_nan_1d_arr(res.origin_torosity.values)
		sns.regplot(np.nan_to_num(res.origin_torosity.values), res.threat_torosity.values, ax=axarr[1], robust=True, n_boot=100, truncate=True, ci=None, 
				line_kws={"color":"red", "linewidth":2,  "label":"Robust lin. reg."}, 
				scatter_kws={"color":"purple", "alpha":.4, "s":150})
		sns.regplot(np.nan_to_num(res.origin_torosity.values), res.threat_torosity.values, ax=axarr[1], robust=False, n_boot=100, truncate=True, scatter=False, ci=None,
				line_kws={"color":"blue", "linewidth":2,  "label":"lin. reg."}, )


		# Plot examples of traces with different torosities
		for th, c, ax, cmap in zip(threshold, colors, axarr[4:], colormaps):
			axarr[0].axvline(th[0], color=c, alpha=.5)
			axarr[0].axvline(th[1], color=c, alpha=.5)

			img = np.ones_like(self.agent.maze)*100
			img[0, 0] = 0
			ax.imshow(img, cmap="Greys_r")

			tor = res.loc[(res[use_tor] <= th[1]) & (res[use_tor] >= th[0])]
			for i,t in tor.iterrows():
				if use_tor == "threat_torosity":
					end = t.time_out_of_t
				else:
					end = -1

				x,y = t.tracking_data[:end, 0, 0], t.tracking_data[:end, 1, 0]
				ax.plot(x,y,  linestyle="-", linewidth=4, color=c, alpha=0.5,label="LOW tor")
				# ax.scatter(x,y, c=np.arange(len(x)), cmap=cmap, vmin=10,  alpha=.9, s=50)

			if use_tor == "threat_torosity": 
				ax.set(title='Example trajectories - th: {}'.format(th), xticks=[], yticks=[], xlim=[100, 150], ylim=[225, 150])
			else:
				ax.set(title='Example trajectories - th: {}'.format(th), xticks=[], yticks=[])

		# Set axes
		axarr[0].set(title="{}".format(use_tor), ylabel="count", xlabel="torosity")
		axarr[1].set(title="Escape arm vs Torosity", xlabel="threat_torosity", yticks=[0,1], yticklabels=["left", "right"])
		axarr[2].set(title='Total vs Threat Torosity', xlabel='overall', ylabel='threat', xlim=[0.75, 5], ylim=[0.75, 5])
		axarr[3].set(title='torosity vs speed', xlabel='mean speed', ylabel='torosity', xlim=[0, 4], ylim=[0, 4])

		for ax in axarr[:4]:
			ax.legend()


	def plot_binned_threat_torosity(self):
		# ? get binned torosity
		# res, ares, res1, ares1, res2, ares2 = self.load_all_res()

		# f, ax = plt.subplots()

		# # loop over each trial
		# all_trials_tors = []
		# for i,tr in res.iterrows():
		# 	print("processing trial ", i)
		# 	tor = tr.threat_torosity
		# 	end = tr.time_out_of_t
		# 	tracking = tr.tracking_data
		# 	br = tr.escape_arm

		# 	# Loop over each time step before the mouse leaves the threat platform
		# 	trial_tor = []
		# 	for i in np.arange(end):
		# 		trial_tor.append(self.process_one_trial(None, br, tracking=tracking[i:end, :, :], goal=list(tracking[end, :2, 0])))

		# 	all_trials_tors.append(trial_tor)

		# 	ax.plot(trial_tor, linewidth=2, alpha=.5)

		# res['time_binned_threat_torosity'] = all_trials_tors

		# res.to_pickle("Processing/trials_analysis/torosity_final.pkl")

		res = pd.read_pickle("Processing/trials_analysis/torosity_final.pkl")
		res = res.sort_values("threat_torosity", )
		f, ax = plt.subplots()

		for i,tbt in enumerate(res.time_binned_threat_torosity):
			try:
				normalised = normalise_to_val_at_idx(tbt, 0)
				below_th = np.where(normalised < .75)[0][0]
				ax.plot(normalised, alpha=.3)
				ax.scatter(below_th, normalised[below_th], c='r')
			except: pass

	





if __name__ == "__main__":
	mazes = ["asymmetric", "symmetric"]
	for m in mazes:
		t = Torosity(m, "all")

		# t.analyse_all(bined_tor=False, threat_tor=True, plot=False, save_plots=True)

		t.inspect_results()

		# t.plot_binned_threat_torosity()

		break



	plt.show()









