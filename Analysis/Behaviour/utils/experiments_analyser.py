import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project

from Utilities.imports import *


import pickle
import pymc3 as pm
import statsmodels.api as sm
from scipy.signal import find_peaks, resample, medfilt

from Modelling.bayes import Bayes
from Modelling.maze_solvers.gradient_agent import GradientAgent
from Modelling.maze_solvers.environment import Environment
from Analysis.Behaviour.utils.trials_data_loader import TrialsLoader
from Analysis.Behaviour.utils.path_lengths import PathLengthsEstimator


"""[This class facilitates the loading of experiments trials data + collects a number of methods for the analysis. ]
"""


class ExperimentsAnalyser(Bayes, Environment, TrialsLoader, PathLengthsEstimator):
	# ! important 
	max_duration_th = 19 # ? only trials in which the mice reach the shelter within this number of seconds are considered escapes (if using escapes == True)
	

	# Variables look ups
	if sys.platform != "darwin": # folder in which the pickled trial data are saved
		metadata_folder = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\Psychometric"
	else:
		metadata_folder = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/analysis_metadata/Psychometric"

	# decision theory simulation params
	speed_mu, speed_sigma = 5, 2.5
	speed_noise = 0
	distance_noise = .1


	def __init__(self, naive=None, lights=None, escapes=None, escapes_dur=None, shelter=True, 
			agent_params=None, **kwargs):
		# Initiate some parent classes
		Bayes.__init__(self) # Get functions from bayesian modelling class
		PathLengthsEstimator.__init__(self)
		
		# store params
		self.naive, self.lights, self.escapes, self.escapes_dur, self.shelter = naive, lights, escapes, escapes_dur, shelter

		# Get trials data
		TrialsLoader.__init__(self, **kwargs)

		# Load geodesic agent
		if agent_params is None:
			self.agent_params = dict(grid_size=1000, maze_design="PathInt2_old.png")
		else: self.agent_params = agent_params
		Environment.__init__(self, **self.agent_params )
		self.maze = np.rot90(self.maze, 2)


	"""
	||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
						 DATA MANIPULATION
	||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
	"""
	def get_binary_trials_per_condition(self, conditions):
		# ? conditions should be a dict whose keys should be a list of strings with the names of the different conditions to be modelled
		# ? the values of conditions should be a a list of dataframes, each specifying the trials for one condition (e.g. maze design) and the session they belong to

		# Parse data
		# Get trials
		trials = {c:[] for c in conditions.keys()}
		for condition, df in conditions.items():
			sessions = sorted(set(df.session_uid.values))
			for sess in sessions:
				trials[condition].append([1 if "right" in arm.lower() else 0 for arm in df.loc[df.session_uid==sess].escape_arm.values])

		# Get hits and number of trials
		hits = {c:[np.sum(t2) for t2 in t] for c, t in trials.items()}
		ntrials = {c:[len(t2) for t2 in t] for c,t in trials.items()}
		p_r = {c: [h/n for h,n in zip(hits[c], ntrials[c])] for c in hits.keys()}
		n_mice = {c:len(v) for c,v in hits.items()}
		return hits, ntrials, p_r, n_mice, trials

	def merge_conditions_trials(self, dfs):
		# dfs is a list of dataframes with the trials for each condition
		merged = dfs[0]
		for df in dfs[1:]:
			merged = merged.append(df)
		return merged

	def get_hits_ntrials_maze_dataframe(self):
		data = dict(id=[], k=[], n=[], maze=[])
		hits, ntrials, p_r, n_mice, _ = self.get_binary_trials_per_condition(self.conditions)

		for i, (k, n) in enumerate(zip(hits.values(), ntrials.values())):
			for ii, (kk, nn) in enumerate(zip(k, n)):
				data["id"].append(ii)
				data["k"].append(kk)
				data["n"].append(nn)
				data["maze"].append(i)

		data = pd.DataFrame.from_dict(data)
		return data



	"""
		||||||||||||||||||||||||||||    DECISION THEORY MODEL     |||||||||||||||||||||
	"""
	def dt_model_speed_and_distances(self, plot=False, sqrt=True):
		speed = stats.norm(loc=self.speed_mu, scale=self.speed_sigma)
		if sqrt: dnoise = math.sqrt(self.distance_noise)
		else: dnoise = self.distance_noise
		distances = {a.maze:stats.norm(loc=a[self.ratio], scale=dnoise) for i,a in self.paths_lengths.iterrows()}
		
		if plot:
			f, axarr = create_figure(subplots=True, ncols=2)
			dist_plot(speed, ax=axarr[0])

			for k,v in distances.items():
				dist_plot(v, ax=axarr[1], label=k)

			for ax in axarr: make_legend(ax)

		return speed, distances

	def simulate_trials_analytical(self):
		# Get simulated running speed and path lengths estimates
		speed, distances = self.dt_model_speed_and_distances(plot=False)

		# right arm
		right = distances["maze4"]

		# Compare each arm to right
		pR = {k:0 for k in distances.keys()}
		for left, d in distances.items():
			# p(R) = phi(-mu/sigma) and mu=mu_l - mu_r, sigma = sigma_r^2 + sigma_l^2
			mu_l, sigma_l = d.mean(), d.std()
			mu_r, sigma_r = right.mean(), right.std()

			mu, sigma = mu_l - mu_r, sigma_r**2 + sigma_l**2
			pR[left] = round(1 - stats.norm.cdf(-mu/sigma,  loc=0, scale=1), 3)
		return pR

	def simulate_trials(self, niters=1000):
		# Get simulated running speed and path lengths estimates
		speed, distances = self.dt_model_speed_and_distances(plot=False, sqrt=False)

		# right arm
		right = distances["maze4"]

		# Compare each arm to right
		trials, pR = {k:[] for k in distances.keys()}, {k:0 for k in distances.keys()}
		for left, d in distances.items():
			# simulate n trials
			for tn in range(niters):
				# Draw a random length for each arms
				l, r = d.rvs(), right.rvs()

				# Draw a random speed and add noise
				if self.speed_noise > 0: s = speed.rvs() + np.random.normal(0, self.speed_noise, size=1) 
				else: s = speed.rvs()

				# Calc escape duration on each arma nd keep the fastest
				# if r/s <= l/s:
				if r <= l:
					trials[left].append(1)
				else: 
					trials[left].append(0)

			pR[left] = np.mean(trials[left])
		return trials, pR
		
	def fit_model(self):
		xp =  np.linspace(.8, 1.55, 200)
		xrange = [.8, 1.55]

		# Get paths length ratios and p(R) by condition
		hits, ntrials, p_r, n_mice = self.get_binary_trials_per_condition(self.conditions)
		
		# Get modes on individuals posteriors and grouped bayes
		modes, means, stds = self.get_hb_modes()
		grouped_modes, grouped_means = self.bayes_by_condition_analytical(mode="grouped", plot=False) 

		# Plot each individual's pR and the group mean as a factor of L/R length ratio
		f, axarr = create_figure(subplots=True, ncols=2)
		ax = axarr[1]
		mseax = axarr[0]
			
		lr_ratios_mean_pr = {"grouped":[], "individuals_x":[], "individuals_y":[], "individuals_y_sigma":[]}
		for i, (condition, pr) in enumerate(p_r.items()):
			x = self.paths_lengths.loc[self.paths_lengths.maze == condition][self.ratio].values
			y = means[condition]

			# ? plot HB PR with errorbars
			ax.errorbar(x, np.mean(y), yerr=np.std(y), 
						fmt='o', markeredgecolor=self.colors[i+1], markerfacecolor=self.colors[i+1], markersize=15, 
						ecolor=desaturate_color(self.colors[i+1], k=.7), elinewidth=3, 
						capthick=2, alpha=1, zorder=0)             

		def residual(distances, sigma):
			self.distance_noise = sigma
			analytical_pr = self.simulate_trials_analytical()
			return np.sum(np.array(list(analytical_pr.values())))
			 
		params = Parameters()
		params.add("sigma", min=1.e-10, max=.5)
		model = Model(residual, params=params)
		params = model.make_params()
		params["sigma"].min, params["sigma"].max = 1.e-10, 1

		ytrue = [np.mean(m) for m in means.values()]
		x = self.paths_lengths[self.ratio].values

		result = model.fit(ytrue, distances=x, params=params)

		# ? Plot best fit
		# best_sigma = sigma_range[np.argmin(mserr)]
		best_sigma = result.params["sigma"].value
		self.distance_noise = best_sigma

		analytical_pr = self.simulate_trials_analytical()
		pomp = plot_fitted_curve(sigmoid, self.paths_lengths[self.ratio].values, np.hstack(list(analytical_pr.values())), ax, xrange=xrange, 
			scatter_kwargs={"alpha":0}, 
			line_kwargs={"color":white, "alpha":1, "lw":6, "label":"model pR - $\sigma : {}$".format(round(best_sigma, 2))})


		# Fix plotting
		ortholines(ax, [1, 0,], [1, .5])
		ortholines(ax, [0, 0,], [1, 0], ls=":", lw=1, alpha=.3)
		ax.set(title="best fit logistic regression", ylim=[-0.01, 1.05], ylabel="p(R)", xlabel="Left path length (a.u.)",
				 xticks = self.paths_lengths[self.ratio].values, xticklabels = self.conditions.keys())
		make_legend(ax)
		
	"""
	||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
						 BAYES
	||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
	"""
	def bayes_by_condition_analytical(self, load=True, mode="grouped", plot=True):
		if not load: raise NotImplementedError
		else:
			data = self.load_trials_from_pickle()
		return self.analytical_bayes_individuals(conditions=None, data=data, mode=mode, plot=plot)

	def bayes_by_condition(self, conditions=None,  load=False, tracefile="a.pkl", plot=True):
		tracename = os.path.join(self.metadata_folder, tracefile)

		if conditions is None:
			conditions = self.conditions

		if not load:
			trace = self.model_hierarchical_bayes(conditions)
			self.save_bayes_trace(trace, tracename)
			trace = pm.trace_to_dataframe(trace)
		else:
			trace = self.load_trace(tracename)

		# Plot by condition
		good_columns = {c:[col for col in trace.columns if col[0:len(c)] == c] for c in conditions.keys()}
		condition_traces = {c:trace[cols].values for c, cols in good_columns.items()}

		if plot:
			f, axarr = plt.subplots(nrows=len(conditions.keys()))
			for (condition, data), color, ax in zip(condition_traces.items(), ["w", "m", "g", "r", "b", "orange"], axarr):
				for i in np.arange(data.shape[1]):
					if i == 0: label = condition
					else: label=None
					sns.kdeplot(data[:, i], ax=ax, color=color, shade=True, alpha=.15)

				ax.set(title="p(R) posteriors {}".format(condition), xlabel="p(R)", ylabel="pdf", facecolor=[.2, .2, .2])
				ax.legend()            
			plt.show()

		return condition_traces

	def get_hb_modes(self, trace=None):
		if trace is None:
			trace = self.bayes_by_condition(conditions=self.conditions, load=True, tracefile="psychometric_individual_bayes.pkl", plot=False)
		
		n_bins = 100
		bins = {k:[np.digitize(a, np.linspace(0, 1, n_bins)) for a in v.T] for k,v in trace.items()}
		modes = {k:[np.median(b)/n_bins for b in bins] for k,bins in bins.items()}
		stds = {k:[np.std(m) for m in v.T] for k,v in trace.items()}
		means = {k:[np.mean(m) for m in v.T] for k,v in trace.items()}

		return modes, means, stds, trace
	
	def closer_look_at_hb(self):
		# Get paths length ratios and p(R) by condition
		hits, ntrials, p_r, n_mice, trials = self.get_binary_trials_per_condition(self.conditions)
		modes, means, stds, traces = self.get_hb_modes()
		f, axarr = create_figure(subplots=False, nrows=4, sharex=True)
		f2, ax2 = create_figure(subplots=False)

		aboves, belows, colors = [], [], []
		for i, (condition, trace) in enumerate(traces.items()):
			sort_idx = np.argsort(means[condition])
			nmice = len(sort_idx)
			above_chance, below_chance = 0, 0
			for mn, id in enumerate(sort_idx):
				tr = trace[:, id]
				# Plot KDE of posterior
				kde = fit_kde(random.choices(tr,k=5000), bw=.025)
				plot_kde(axarr[i], kde, z=mn, vertical=True, normto=.75, color=self.colors[i+1], lw=.5)

				# plot 95th percentile_range of posterior's means
				percrange = percentile_range(tr)
				axarr[i].scatter(mn, percrange.mean, color=self.colors[i+1], s=25)
				axarr[i].plot([mn, mn], [percrange.low, percrange.high], color=white, lw=2, alpha=1)

				if percrange.low > .5: above_chance += 1
				elif percrange.high < .5: below_chance += 1

			axarr[i].text(0.95, 0.1, '{}% above .5 - {}% below .5'.format(round(above_chance/nmice*100, 2), round(below_chance/nmice*100, 2)), color=grey, fontsize=15, transform=axarr[i].transAxes, **text_axaligned)
			aboves.append(round(above_chance/nmice*100, 2))
			belows.append(round(below_chance/nmice*100, 2))
			colors.append(self.colors[i+1])

			axarr[i].set(ylim=[0, 1], ylabel=condition)
			axarr[i].axhline(.5, **grey_dotted_line)
		
		axarr[0].set(title="HB posteriors")
		axarr[-1].set(xlabel="mouse id")

		sns.barplot( np.array(aboves), np.arange(4)+1, palette=colors, orient="h", ax=ax2)
		sns.barplot(-np.array(belows), np.arange(4)+1, palette=colors, orient="h", ax=ax2)
		ax2.axvline(0, **grey_line)
		ax2.set(title="Above and below chance posteriors%", xlabel="%", ylabel="maze id", xlim = [-20, 80])

	def inspect_hbv2(self):
		# Plot the restults of the alternative hierarchical model

		# Load trace and get data
		trace = self.load_trace(os.path.join(self.metadata_folder, "test_hb_trace.pkl"))
		data = self.get_hits_ntrials_maze_dataframe()


		# Plot
		f, axarr = create_figure(subplots=True, ncols=2, nrows=2)

		xmaxs = {k:[[], []] for k,v in self.conditions.items()}
		for column in trace:
			if not "beta_theta" in column: continue
			n = int(column.split("_")[-1])
			mouse = data.iloc[n]
			ax = axarr[mouse.maze]
			color = self.colors[1 + mouse.maze]

			kde = fit_kde(random.choices(trace[column], k=5000),   bw=.025)
			plot_kde(ax, kde, color=desaturate_color(color), z=0, alpha=0)

			xmax = kde.support[np.argmax(kde.density)]
			vline_to_curve(ax, xmax, kde.support, kde.density, dot=True, 
							line_kwargs=dict(alpha=.85, ls="--", lw=2, color=color), scatter_kwargs=dict(s=100, color=color), zorder=100)
			ax.scatter(xmax, 0, s=100, color=color, alpha=.8)
			xmaxs["maze{}".format(mouse.maze+1)][0].append(xmax)
			xmaxs["maze{}".format(mouse.maze+1)][1].append(np.std(kde.density))

		for i, (k, v) in enumerate(xmaxs.items()):
			# kde = fit_kde(v[0], kernel="gau", bw=.025)
			kde = fit_kde(v[0],  weights=np.array(v[1]), kernel="gau", fft=False, bw=.025, gridsize=250) 
			plot_kde(axarr[i], kde, invert=True, color=self.colors[i+1], z=0, alpha=.2)
			axarr[i].axhline(0, **grey_line)


		axarr[2].set(title="maze 3", xlim=[-.1, 1.1], ylim=[-8, 8])
		axarr[1].set(title="maze 2", xlim=[-.1, 1.1], ylim=[-8, 8])
		axarr[3].set(title="maze 4", xlim=[-.1, 1.1], ylim=[-8, 8])
		axarr[0].set(title="maze 1", xlim=[-.1, 1.1], ylim=[-8, 8])


	"""
	||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
						 REACTION TIME
	||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
	"""
	def inspect_rt_metric(self, load=False, plot=True):
		def get_first_peak(x, **kwargs):
			peaks, _ = find_peaks(x, **kwargs)
			if not np.any(peaks): return 0
			return peaks[0]

		def get_above_th(x, th):
			peak = 0
			while peak <= 0 and th>0:
				try:
					peak = np.where(x > th)[0][0]
				except:
					peak =  0
				th -= .1
			return peak

		if not load:
			# ? th def
			bodyth, snouth, rtth = 6, 6, 2.5

			data = self.merge_conditions_trials(list(self.conditions.values()))
			data = data.loc[data.is_escape == "true"]
			if plot:
				f = plt.subplots(sharex=True)

				grid = (2, 5)
				mainax = plt.subplot2grid(grid, (0, 0), colspan=2, rowspan=2)
				bodyax = plt.subplot2grid(grid, (0, 2), colspan=2)
				snoutax = plt.subplot2grid(grid, (1, 2), colspan=2)
				scatterax = plt.subplot2grid(grid, (0, 4))
				histax = plt.subplot2grid(grid, (1, 4))

				mainax.set(title="$speed_{snout} - speed_{body}$", ylim=[-6, 12], xlabel="time (frames)", ylabel="ratio", xlim=[0, 80])
				bodyax.set(ylim=[-2.1, 15], title="body",  xlabel="time (frames)", ylabel="speed (a.u.)", xlim=[0, 80])
				snoutax.set(ylim=[-2.1, 15], title="snout", xlabel="time (frames)", ylabel="speed (a.u.)", xlim=[0, 80])
				scatterax.set(xlabel="body peaks", ylabel="snout peaks", xlim=[0, 120], ylim=[0, 120])
				histax.set(title="reaction times", xlabel="time (s)", ylabel="density")

			datadict = {"trialid":[], "rt_frame":[], "fps":[], "rt_s":[], "rt_frame_originalfps":[]}
			bodypeaks, snoutpeaks, rtpeaks = [], [], []
			for n, (i, trial) in tqdm(enumerate(data.iterrows())):
				# Get data
				start, end = trial.stim_frame, int(np.ceil(trial.stim_frame + (trial.time_out_of_t * trial.fps)))
				
				if end-start < 5: continue

				allsess_body = (TrackingData.BodyPartData & "bpname='body'" & "recording_uid='{}'".format(trial.recording_uid)).fetch("tracking_data")[0]
				allsess_snout = (TrackingData.BodyPartData & "bpname='snout'" & "recording_uid='{}'".format(trial.recording_uid)).fetch("tracking_data")[0]
				body, snout = allsess_body[start:end, :].copy(), allsess_snout[start:end, :].copy()

				# ? remove tracking errors
				body[:, 2][np.where(body[:, 2] > 25)] = np.nan
				snout[:, 2][np.where(snout[:, 2] > 25)] = np.nan

				# If filmed at 30fps, upsample
				if trial.fps < 40:
					new_n_frames = np.int(body.shape[0] / 30 * 40)
					body = resample(body, new_n_frames)
					snout = resample(snout, new_n_frames) 

					body[:, 2], snout[:, 2] = body[:, 2]/30*40, snout[:, 2]/30*40

				# Get speeds
				bs, ss = line_smoother(body[:, 2],  window_size=11, order=3,), line_smoother(snout[:, 2],  window_size=11, order=3,)
				rtmetric = ss-bs

				# Get first peak
				bpeak, speak, rtpeak = get_above_th(bs, bodyth), get_above_th(ss, snouth), get_above_th(rtmetric, rtth)

				# Append to dictionary
				if bpeak > 0  and speak > 0:
					rt = rtpeak
				elif bpeak == 0 and speak > 0:
					rt = speak
				elif bpeak > 0 and speak == 0:
					rt = bpeak
				else:
					rt = None

				#  data = {"trialid":[], "rt_frame":[], "fps":[], "rt_s":[], "rt_frame_originalfps":[]}
				if rt is not None:
					datadict["trialid"].append(trial.trial_id)
					datadict["rt_frame"].append(rt)
					datadict["fps"].append(trial.fps)
					datadict["rt_s"].append(rt/40)
					datadict["rt_frame_originalfps"].append(int(np.ceil(rt/40 * trial.fps)))
				else:
					datadict["trialid"].append(trial.trial_id)
					datadict["rt_frame"].append(np.nan)
					datadict["fps"].append(trial.fps)
					datadict["rt_s"].append(np.nan)
					datadict["rt_frame_originalfps"].append(np.nan)
				
				# Append to listdir
				bodypeaks.append(bpeak)
				snoutpeaks.append(speak)
				rtpeaks.append(rtpeak)

				# Plot
				if plot:
					bodyax.plot(bs, color=green, alpha=.2)
					bodyax.scatter(bpeak, bs[bpeak], color=green, alpha=1)

					snoutax.plot(ss, color=red, alpha=.2)
					snoutax.scatter(speak, ss[speak], color=red, alpha=1)

					mainax.plot(rtmetric, color=magenta, alpha=.2)
					mainax.scatter(rtpeak, rtmetric[rtpeak], color=magenta, alpha=1)

					scatterax.scatter(bpeak, speak, color=white, s=100, alpha=.4)

			if plot:
				scatterax.plot([0, 200], [0, 200], **grey_line)
				mainax.axhline(rtth, **grey_line)
				snoutax.axhline(snouth, **grey_line)
				bodyax.axhline(bodyth, **grey_line)

				# Plot KDE of RTs
				kde = sm.nonparametric.KDEUnivariate(datadict["rt_s"])
				kde.fit(bw=.1) # Estimate the densities
				histax.fill_between(kde.support, 0, kde.density, alpha=.2, color=lightblue, lw=3,zorder=10)
				histax.plot(kde.support, kde.density, alpha=1, color=lightblue, lw=3,zorder=10)

				# Plot KDE of the peaks 
				for i, (ax, peaks, color) in enumerate(zip([mainax, snoutax, bodyax], [rtpeaks, snoutpeaks, bodypeaks], [magenta, red, green])):
					kde = sm.nonparametric.KDEUnivariate(np.array(peaks).astype(np.float))
					kde.fit(bw=1) # Estimate the densities
					if i == 0:
						x, y, z = kde.support, kde.density/np.max(kde.density)*2 - 6, -6
					else:
						x, y, z = kde.support, kde.density/np.max(kde.density)*1.5 - 2, -2
					ax.fill_between(x, z, y, alpha=.2, color=color, lw=3,zorder=10)
					ax.plot(x, y, alpha=1, color=color, lw=3,zorder=10)

			# Save to pickle
			datadf = pd.DataFrame.from_dict(datadict)
			datadf.to_pickle(os.path.join(self.metadata_folder, "reaction_time.pkl"))
		else:
			datadf = pd.read_pickle(os.path.join(self.metadata_folder, "reaction_time.pkl"))
		return datadf


	"""
		||||||||||||||||||||||||||||    TIMED ANALYSIS    |||||||||||||||||||||
	"""

	def plot_effect_of_time(self, xaxis_istime=True, robust=False):
		def print_lin_reg(x, y, ax):
			_, p1, p2, res = linear_regression(x, y)
			ax.text(0.95, 0.8, 'slope:{} - $r^2${}'.format(round(p2, 5), round(res.rsquared, 2)), color="k", fontsize=15, transform=ax.transAxes, **text_axaligned)

		rtdf = self.inspect_rt_metric(load=True, plot=False)

		if xaxis_istime: bw = 60
		else: bw = 0.5

		# crate figure
		f, axarr = create_figure(subplots=True, nrows=5, ncols=3, sharex=True)
		leftcol, centercol, rightcol = [0, 3, 6, 9, 12], [1, 4, 7, 10, 13], [2, 5, 8, 11, 14]

		# loop over experiments
		all_speeds, all_rts, all_times, all_times2 = [], [], [], []
		for i, (condition, df) in enumerate(self.conditions.items()):
			ax = axarr[leftcol[i]]
			axspeed = axarr[centercol[i]]
			axrt = axarr[rightcol[i]]

			times, times2, ones, zeros, speeds, rts, = [], [], [], [], [], [],
			# loop over trials
			for n, (_, trial) in enumerate(df.iterrows()):
				# get time of trial and escape arm
				if xaxis_istime:
					x = trial.stim_frame_session / trial.fps
				else:
					x = trial.trial_number

				if 'Right' in trial.escape_arm:
					y = 1
					ones.append(x)
				else:
					y = 0
					zeros.append(x)

				# Get escape speed
				# if y == 1:
					# escape_speed = np.percentile(line_smoother(trial.tracking_data[:, 2], window_size=51, order=5), 80) / trial.fps
				escape_speed = np.mean(line_smoother(trial.tracking_data[:, 2], window_size=51, order=5)) / trial.fps
				times.append(x)
				speeds.append(escape_speed)

				# Get reaction time
				rt = rtdf.loc[rtdf.trialid == trial.trial_id].rt_s.values
				if np.any(rt): 
					if not np.isnan(rt[0]):
						times2.append(x)
						rts.append(rt[0])

				# plot
				ax.scatter(x, y, color=colors[i], s=50, alpha=.5)

			# linear regression on speed and rt
			try:
				sns.regplot(times, speeds, ax=axspeed, robust=robust, scatter=True, order=1, scatter_kws=dict(s=25, color=desaturate_color(self.colors[i+1], k=.8)),
							line_kws=dict(color=self.colors[i+1], lw=2, alpha=1), truncate=True,)
				print_lin_reg(times, speeds, axspeed)

				sns.regplot(times2, rts, ax=axrt, robust=robust, scatter=True, order=1, scatter_kws=dict(s=25, color=desaturate_color(self.colors[i+1], k=.8)),
							line_kws=dict(color=self.colors[i+1], lw=2, alpha=1), truncate=True,)
				print_lin_reg(times2, rts, axrt)
					
				# store data
				all_speeds.extend(speeds)
				all_rts.extend(rts)
				all_times.extend(times)
				all_times2.extend(times2)
			except:
				continue

			# Plot KDEs
			ax, kde_right = plot_kde(ax, fit_kde(ones, bw=bw), .8, invert=True, normto=.25, color=self.colors[i+1])
			ax, kde_left = plot_kde(ax, fit_kde(zeros, bw=bw), .2, invert=False, normto=.25, color=self.colors[i+1])

			# Plot ratio of KDEs in last plot
			xxx = np.linspace(np.max([np.min(kde_right.support), np.min(kde_left.support)]), np.min([np.max(kde_right.support), np.max(kde_left.support)]), 1000)
			ratio = [kde_right.evaluate(xx)/(kde_right.evaluate(xx)+kde_left.evaluate(xx)) for xx in xxx]
			axarr[leftcol[4]].plot(xxx, ratio, lw=3, color=self.colors[i+1], label=condition)

		sns.regplot(all_times, all_speeds, ax=axarr[centercol[i+1]], robust=robust, scatter=True, order=1, scatter_kws=dict(s=25, color=[.3, .3, .3]),
					line_kws=dict(color="k", lw=2, alpha=1), truncate=True,)

		print_lin_reg(all_times, all_speeds, axarr[centercol[i+1]])

		
		sns.regplot(all_times2, all_rts, ax= axarr[rightcol[i+1]], robust=robust, scatter=True, order=1, scatter_kws=dict(s=25, color=[.3, .3, .3]),
					line_kws=dict(color="k", lw=2, alpha=1), truncate=True,)
		print_lin_reg(all_times2, all_rts, axarr[rightcol[i+1]])

		# Set axes correctly
		for i, ax in enumerate(axarr):
			if i in leftcol:
				kwargs = dict(ylim=[-.1, 1.1], yticklabels=["left", "right"], ylabel="escape", yticks=[0, 1],  ) 
			elif i in centercol:
				kwargs = dict(ylabel="mean speed", ylim=[0, .5])
			else:
				kwargs = dict(ylabel="rt (s)", ylim=[0, 13])

			if xaxis_istime:
				ax.set(xticks=[x*60 for x in np.linspace(0, 100, 11)], xticklabels=np.linspace(0, 100, 11), **kwargs)
			else:
				ax.set(xlim=[0, 20], **kwargs)

		axarr[leftcol[0]].set(title="Left/Right")
		axarr[centercol[0]].set(title="Escape speed")
		axarr[rightcol[0]].set(title="Reaction time")

		if xaxis_istime:
			xlab = "time (min)"
		else:
			xlab = "trial #"

		axarr[leftcol[-1]].set(xlabel=xlab)
		axarr[centercol[-1]].set(xlabel=xlab)
		axarr[rightcol[-1]].set(title="Reaction times", xlabel=xlab)

		axarr[leftcol[4]].set(title="balance over time", xlabel=xlab, ylabel="R / L+R")
		make_legend(axarr[leftcol[4]])
		make_legend(axarr[rightcol[-1]])

	def timed_pr(self):
		grouped_modes, grouped_means, grouped_params, _, _ = self.bayes_by_condition_analytical(mode="grouped", plot=False) 

		# crate figure
		f, axarr = create_figure(subplots=True, nrows=4,sharex=True, sharey=True)

		window_size = 600
		magification_factor = 100
		n_steps = 80

		# loop over experiments
		for i, (condition, trials) in enumerate(self.conditions.items()):
			ax = axarr[i]

			# Get escape arms by time
			trial_times = trials.stim_frame_session.values / trials.fps.values
			trial_outcomes = np.array([1 if "Right" in t.escape_arm else 0 for i,t in trials.iterrows()])

			trial_outcomes = trial_outcomes[np.argsort(trial_times)]
			trial_times = np.sort(trial_times)

			# Sweep over time and do windowed analytical bayes
			means, std = [], []

			for t in np.linspace(np.min(trial_times), np.max(trial_times), n_steps):
				in_window = np.where((trial_times > t-window_size/2) & (trial_times < t+window_size/2))
				if np.any(in_window):
					outcomes_in_window = trial_outcomes[in_window]

					if len(outcomes_in_window) < 5: continue # skip times when there are too few trials
					
					(a, b, fact), mean, var = self.simple_analytical_bayes(outcomes_in_window)

					beta, support, density = get_parametric_distribution("beta", a, b, x0=0.05, x1=0.95)
					try:
						ax.plot(density/np.max(density)*magification_factor+t, support, alpha=1, color=self.colors[i+1])
					except:
						pass

					# ortholines(ax, [1, 1], [t-window_size/2, t+window_size/2], ls=":", lw=1, color=grey)

			ax.axhline(grouped_means[condition], color=self.colors[i+1], ls="--", lw=1)
			ortholines(ax, [0, 0], [0, 1], ls=":", lw=1, color=grey)
			ax.set(ylim=[-0.1, 1.1],  xticks=[x*60 for x in np.linspace(0, 100, 11)], xticklabels=np.linspace(0, 100, 11), ylabel=condition)


		axarr[0].set(title="timed grouped bayesian")
		axarr[-1].set(xlabel="time (min)",)

	"""
	||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
						 PLOT P(R)
	||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
	"""
	def plot_pr_vs_time(self):
		# loop over different settings for kde
		for bw in [60]:
			# crate figure
			f, axarr = create_figure(subplots=True, nrows=5, ncols=2, sharex=True)
			leftcol, rightcol = [0, 2, 4, 6, 8], [1, 3, 5, 7, 9]

			# loop over experiments
			for i, (condition, df) in enumerate(self.conditions.items()):
				ax = axarr[leftcol[i]]
				axspeed = axarr[rightcol[i]]
				times, ones, zeros, speeds = [], [], [], []
				# loop over trials
				for n, (_, trial) in enumerate(df.iterrows()):
					# get time of trial and escape arm
					x = trial.stim_frame_session / trial.fps
					if 'Right' in trial.escape_arm:
						y = 1
						ones.append(x)
					else:
						y = 0
						zeros.append(x)

					# Get escape speed
					escape_speed = np.mean(trial.tracking_data[:, 2]) / trial.fps
					times.append(x)
					speeds.append(escape_speed)

					# plot
					ax.scatter(x, y, color=self.colors[i+1], s=50, alpha=.5)

				# linear regression on speed and
				sns.regplot(times, speeds, ax=axspeed, scatter=True, order=1, scatter_kws=dict(s=75, color=desaturate_color(self.colors[i+1], k=.5)),
								line_kws=dict(color=self.colors[i+1], lw=2, alpha=1), truncate=True,)

				# Plot KDEs
				ax, kde_right = plot_kde(ax, fit_kde(ones, bw=bw), .8, invert=True, normto=.25, color=self.colors[i+1])
				ax, kde_left = plot_kde(ax, fit_kde(zeros, bw=bw), .2, invert=False, normto=.25, color=self.colors[i+1])

				# Plot ratio of KDEs in last plot
				xxx = np.linspace(np.max([np.min(kde_right.support), np.min(kde_left.support)]), np.min([np.max(kde_right.support), np.max(kde_left.support)]), 1000)
				ratio = [kde_right.evaluate(xx)/(kde_right.evaluate(xx)+kde_left.evaluate(xx)) for xx in xxx]
				axarr[leftcol[4]].plot(xxx, ratio, lw=3, color=self.colors[i+1], label=condition)

			# Set axes correctly
			for i, (ax, maze) in enumerate(zip(axarr, list(self.conditions.keys()))):
				if i in leftcol:
					kwargs = dict( ylim=[-.1, 1.1], yticklabels=["left", "right"], ylabel="escape", yticks=[0, 1],  ) 
				else:
					kwargs = dict(ylabel="90th perc of escape speed (a.u)")

				ax.set(title=maze, xlabel="time (min)",  xticks=[x*60 for x in np.linspace(0, 100, 11)], xticklabels=np.linspace(0, 100, 11), **kwargs)

			ortholines(axarr[4], [0,], [.5])
			axarr[leftcol[4]].set(title="balance over time", xlabel="time (min)", ylabel="R / (R+L)")
			make_legend(axarr[4])
			f.tight_layout()


	"""
		||||||||||||||||||||||||||||    PLOTTERS     |||||||||||||||||||||
	"""

	def pr_by_condition(self, exclude_experiments=[None], ax=None):
		xp =  np.linspace(.8, 1.55, 200)
		xrange = [.8, 1.55]

		# Get paths length ratios and p(R) by condition
		hits, ntrials, p_r, n_mice, trials = self.get_binary_trials_per_condition(self.conditions)
		
		# Get modes on individuals posteriors and grouped bayes
		modes, means, stds, _ = self.get_hb_modes()
		grouped_modes, grouped_means, grouped_params, sigmasquared, pranges = self.bayes_by_condition_analytical(mode="grouped", plot=False) 

		 # Plot each individual's pR and the group mean as a factor of L/R length ratio
		if ax is None: 
			f, ax = create_figure(subplots=False)
		else: f = None
		
		colors_helper = MplColorHelper("Purples", 0, 5, inverse=True)
		colors = [colors_helper.get_rgb(i) for i in range(len(p_r.keys()))]

		lr_ratios_mean_pr = {"grouped":[], "individuals_x":[], "individuals_y":[], "individuals_y_sigma":[]}
		yticks=[0]
		for i, (condition, pr) in enumerate(p_r.items()):
			if condition not in grouped_modes.keys(): continue 

			yticks.append(grouped_modes[condition])
			x = self.paths_lengths.loc[self.paths_lengths.maze == condition]["distance"].values

			y = means[condition]
			# ? plot HB PR with errorbars
			ax.errorbar(x, grouped_means[condition], yerr=(pranges[condition].low - pranges[condition].high)/2, 
						fmt='o', markeredgecolor=colors[i], markerfacecolor=colors[i], markersize=25, 
						ecolor=desaturate_color(colors[i], k=.7), elinewidth=12, label=condition,
						capthick=2, alpha=1, zorder=20)             
			vline_to_point(ax, x, grouped_modes[condition], color=colors[1], lw=5, ls="--", alpha=.8)
			hline_to_point(ax, x, grouped_modes[condition], color=black, lw=5, ls="--", alpha=.8)


			if condition not in exclude_experiments:# ? use the data for curves fitting
				k = .4
				lr_ratios_mean_pr["grouped"].append((x[0], np.mean(pr), np.std(pr)))  
				lr_ratios_mean_pr["individuals_x"].append([x[0] for _ in np.arange(len(y))])
				lr_ratios_mean_pr["individuals_y"].append(y)
				lr_ratios_mean_pr["individuals_y_sigma"].append(stds[condition])
			else: 
				del grouped_modes[condition], grouped_means[condition], sigmasquared[condition]
		yticks.append(1)

		# Fix plotting
		ax.set(ylim=[0, 1], xlim=[0, 1000], ylabel="$p(R)$", title=None, xlabel="$L$",
				xticks = self.paths_lengths["distance"].values, xticklabels = ["${}$".format(np.int(x)) for x in self.paths_lengths["distance"].values], 
				yticks=np.array(yticks), yticklabels=["${}$".format(round(y, 2)) for y in yticks])
		# make_legend(ax)
		sns.despine(fig=f, offset=10, trim=False, left=False, right=True)

		return lr_ratios_mean_pr, grouped_modes, grouped_means, sigmasquared, modes, means, stds, f, ax, xp, xrange, grouped_params

	def plot_pr_by_condition_detailed(self):
		for bw in [0.02]:
			f, axarr = create_figure(subplots=True, ncols=5, sharey=False)
			# plot normal pR
			lr_ratios_mean_pr, grouped_modes, grouped_means, sigmasquared,  modes, means, stds, _, ax, xp, xrange, grouped_params = self.pr_by_condition(ax=axarr[0])

			# Plot a kde of the pR of each mouse on each maze
			for i, (maze, prs) in enumerate(means.items()):
				if len(prs) == 0: continue
				bins = np.linspace(0, 1, 15)

				# Plot KDE of mice's pR + max density point and horizontal lines
				kde = fit_kde(prs, bw=bw)
				shift = (4-i)*.1-.1

				# Plot KDE of each experiments rpr
				xx, yy = (kde.density*bw)+shift, kde.support
				axarr[1].scatter(np.ones(len(prs))*shift, prs, color=desaturate_color(self.colors[i+1]), s=50)
				plot_shaded_withline(axarr[1],xx, yy, z=shift, color=self.colors[i+1], lw=3, label=maze, zorder=10 )

				# Plot mean and 95th percentile range of probabilities 
				plot_shaded_withline(axarr[2], kde.cdf, yy, color=desaturate_color(self.colors[i+1]), lw=3,  zorder=10 )

				# percrange = percentile_range(prs)
				# axarr[1].scatter(shift, percrange.mean, color=white, s=50, zorder=50)
				# axarr[1].plot([shift, shift], [percrange.low, percrange.high], color=grey, lw=4, alpha=.9, zorder=40)

				# Plot a line to the highest density point
				hmax, yyy = np.max(xx), yy[np.argmax(xx)]
				hline_to_curve(axarr[1], yyy, xx, yy, color=self.colors[i+1], dot=True, line_kwargs=dict(alpha=.5, ls="--", lw=3), scatter_kwargs=dict(s=100))
				axarr[0].axhline(np.mean(prs), color=self.colors[i+1], alpha=.5, ls="--", lw=3)

				# Plot beta distributions of grouped analytical bayes
				beta, support, density = get_parametric_distribution("beta", *grouped_params[maze] )
				plot_shaded_withline(axarr[3], density, support, z=None, color=self.colors[i+1])
				hline_to_curve(axarr[3], support[np.argmax(density)], density, support, color=self.colors[i+1], dot=True, line_kwargs=dict(alpha=.5, ls="--", lw=3), scatter_kwargs=dict(s=100))

				axarr[4].bar(4-i, len(prs), color=self.colors[i+1])

			ortholines(axarr[1], [0,], [.5])
			axarr[1].set(title="individuals pR distribution - bw{}".format(bw), xlabel="probability",  ylim=[0, 1], xlim=[-0.01, 0.5])
			make_legend(axarr[1])

			axarr[2].set(title="individuals pR cdf", xlabel="cdf",  ylim=[0, 1], xlim=[0, 1])


			ortholines(axarr[2], [0,], [.5])
			axarr[3].set(title="grouped bayes", ylabel="p(R)", xlabel="probability",  ylim=[0, 1], xlim=[-2, 20])

		axarr[4].set(title="mice x maze", xticks=[1, 2, 3, 4], xticklabels=["m4", "m3", "m2", "m1"], ylabel="# mice", xlabel="maze")

	def model_summary(self, exclude_experiments=[None], ax=None):
		sns.set_context("talk", font_scale=4.5)

		sns.set_style("white", {
					"axes.grid":"False",
					"ytick.right":"False",
					"ytick.left":"True",
					"xtick.bottom":"True",
					"text.color": "0"
		})

		lr_ratios_mean_pr, grouped_modes, grouped_means, sigmasquared,  modes, means, stds, f, ax, xp, xrange, _ = self.pr_by_condition(exclude_experiments=exclude_experiments, ax=ax)

		# ? Fit logistic regression to mean p(R)+std(p(R))
		# ? Plot sigmoid filled to psy - mean pR of grouped bayes + std
		xdata, ydata = [m[0] for m in lr_ratios_mean_pr["grouped"]], list(grouped_modes.values())

		colors_helper = MplColorHelper("Purples", 0, 5, inverse=True)
		colors = [colors_helper.get_rgb(i) for i in range(len(xdata))]

		pomp = plot_fitted_curve(centered_logistic, xdata, ydata, ax, 
			xrange=[0, 1000],
			fit_kwargs={"sigma":[math.sqrt(s) for s in list(sigmasquared.values())], 
							"method":"dogbox", "bounds":([0.99, 585, 0.01],[1, 595, 0.3])},
			scatter_kwargs={"alpha":0, "c":colors}, 
			line_kwargs={"color":black, "alpha":.85, "lw":10,})

		rhos = [m[0] for m in lr_ratios_mean_pr["grouped"]]
		labels = ["$Maze\\ {}$\n $\\rho = {}$".format(1+i, round(r, 2)) for i,r in enumerate(rhos)]
		# Fix plotting
		# ax.axhline(.5, ls="--", color=grey, lw=.25)
		# ax.set(ylim=[0, 1], ylabel="p(R)", title=None, xlabel="$\\rho$",
		# 		 xticks = self.paths_lengths[self.ratio].values, xticklabels = labels)
		# make_legend(ax)


		sns.despine(offset=10, trim=False, left=False, right=True)
		print(sns.axes_style())

	def plot_hierarchical_bayes_effect(self):
		# Get hierarchical Bayes modes and individual mice p(R)
		hits, ntrials, p_r, n_trials, _ = self.get_binary_trials_per_condition(self.conditions)
		trace = self.bayes_by_condition(conditions=self.conditions, load=True, tracefile="psychometric_individual_bayes.pkl", plot=False) 

		# Get data from the alternative hierarchical model
		v2trace = self.load_trace(os.path.join(self.metadata_folder, "test_hb_trace.pkl"))
		v2_cols = [k  for k in v2trace.columns if "beta_theta" in k]
		data = self.get_hits_ntrials_maze_dataframe()

		# Get the mode of the posteriors
		modes, means, stds, _ = self.get_hb_modes()

		f, axarr = plt.subplots(ncols=4, sharex=True, sharey=True)
		
		for i, (exp, ax) in enumerate(zip(trace.keys(), axarr)):
			# Plot mean and errorbar for naive and standard hb

			ax.errorbar(0.1, np.mean(p_r[exp]), yerr=np.std(p_r[exp]),  **white_errorbar)
			ax.errorbar(0.9, np.mean(means[exp]), yerr=np.std(means[exp]),  **white_errorbar)

			# Plot data from naive and standard hb
			ax.scatter(np.random.normal(0, .025, size=len(p_r[exp])), p_r[exp], color=self.colors[i+1], alpha=.5, s=200)
			ax.scatter(np.random.normal(1, .025, size=len(means[exp])), means[exp], color=self.colors[i+1], alpha=.5, s=200)

			# Plot KDEs
			kde = fit_kde(p_r[exp],   bw=.05)
			plot_kde(ax, kde, z=0, vertical=True, normto=.3, color=self.colors[i+1],)

			kde = fit_kde(means[exp],   bw=.05)
			plot_kde(ax, kde, z=1, vertical=True, invert=False, normto=.3, color=self.colors[i+1],)

			# Add results from alternative hb
			exp_mice = data.loc[data.maze == int(exp[-1])-1].index
			exp_traces_cols = [k for k in v2_cols if int(k.split("_")[-1]) in exp_mice]
			t = v2trace[exp_traces_cols].values
			v2means, v2stds = np.mean(t, 0), np.std(t, 0)

			ax.errorbar(1.9, np.mean(v2means), yerr=np.std(v2means),  **white_errorbar)
			ax.scatter(np.random.normal(2, .025, size=len(v2means)), v2means, color=self.colors[i+1], alpha=.5, s=200)

			kde = fit_kde(v2means,   bw=.05)
			plot_kde(ax, kde, z=2, vertical=True, invert=False, normto=.3, color=self.colors[i+1],)

			# Set ax
			ortholines(ax, [0,], [.5])
			ax.set(title=exp, xlim=[-.1, 3.1], ylim=[-.02, 1.02], xticks=[0, 1, 2], xticklabels=["Raw", "hb means", "hb_v2"], ylabel="p(R)")

	def plot_escape_duration_by_arm(self):
		def get_escape_distance(df, key=None):
			if key is not None:
				df = df.loc[df.maze == key]
			distance = [np.sum(r.tracking_data[:, 2]) for i,r in df.iterrows()]
			return distance

		def get_mean_escape_speed(df, key=None):
			if key is not None:
				df = df.loc[df.maze == key]
			distance = [np.mean(r.tracking_data[:, 2]) for i,r in df.iterrows()]
			return distance			

		for maze, df in self.conditions.items():
			df["maze"] = maze
		escapes = pd.concat([df.loc[df["is_escape"]=="true"] for df in self.conditions.values()])

		# resc, lescs = escapes.loc[escapes.escape_arm == "Right_Medium"], escapes.loc[escapes.escape_arm != "Right_Medium"]
		
		arms = sorted(set(escapes.maze))
		durations, mean_speeds, distances = {arm:[] for arm in arms}, {arm:[] for arm in arms}, {arm:[] for arm in arms}

		for arm in arms:
			durations[arm].extend(list(escapes.loc[escapes.maze == arm].escape_duration))		
			mean_speeds[arm].extend(get_mean_escape_speed(escapes, arm))
			distances[arm].extend(get_escape_distance(escapes, arm))   


		f, ax = plt.subplots()
		colors = MplColorHelper("tab10", 0, 5, inverse=True)

		all_durations = np.hstack(durations.values())
		all_speeds = np.hstack(mean_speeds.values())
		all_distances = np.hstack(distances.values())

		colormap = plt.cm.Reds #or any other colormap
		normalize = mpl.colors.Normalize(vmin=0, vmax=np.max(all_speeds))
		ax.scatter(all_distances, all_durations, c=all_speeds, s=300, cmap="Reds", norm=normalize, edgecolors=black)
		
		ax.set(ylabel="$duration (s)$", xlabel="$distance (a.u.)$", xlim=[0, 1400], ylim=[0, 12],
				xticks=np.arange(0, 1400, 200), xticklabels=["${}$".format(x) for x in np.arange(0, 1400, 200)],
				yticks=np.arange(0, 12, 2), yticklabels=["${}$".format(y) for y in np.arange(0, 12, 2)])

		sns.despine(fig=f, offset=10, trim=False, left=False, right=True)

	"""
		||||||||||||||||||||||||||||    THREAT PLTFORM ANALYSIS     |||||||||||||||||||||
	"""

	def prep_tplatf_trials_data(self, filt=False, fwindow=31, remove_errors=False, speed_th=None):
		"""
			Get the trials for all conditions and get, the tracking specific to the threat, 
			the time at which the mouse leaves T, the speed at which the mouse leaves T

			if true, threat tracking data are passed through a median filter to eliminate errors
		"""

		def cleanup(d, speed_th):
			bad_idx = np.where((d[:, 0] >= 530) & (d[:, 1] <= 255))[0]
			d[bad_idx] = np.nan

			if speed_th is not None:
				fast_idx = np.where(d[:, 2] >= speed_th)
				d[fast_idx] = np.nan
			return d

		self.trials={}
		for condition, data in tqdm(self.conditions.items()):
			out_of_ts, threat_trackings, s_threat_trackings, t_threat_trackings, n_threat_trackings, speeds_at_out_t = [], [], [], [], [], []
			maze_ids = []

			trials = self.get_sessions_trials(maze_design=int(condition[-1]), naive=None, lights=1, escapes=True, escapes_dur=True)
			
			for i, trial in trials.iterrows():
				out_of_t = np.int(trial.time_out_of_t*trial.fps)
				if not filt:
					tracking = trial.tracking_data[:out_of_t, :3].copy()
					s_tracking = trial.snout_tracking_data[:out_of_t, :3].copy()
					n_tracking = trial.neck_tracking_data[:out_of_t, :3].copy()
					t_tracking = trial.tail_tracking_data[:out_of_t, :3].copy()
				else:
					tracking = medfilt(trial.tracking_data[:out_of_t, :3].copy(), [fwindow, 1])
					s_tracking = medfilt(trial.snout_tracking_data[:out_of_t, :3].copy(), [fwindow, 1])
					t_tracking = medfilt(trial.tail_tracking_data[:out_of_t, :3].copy(), [fwindow, 1])
					n_tracking = trial.neck_tracking_data[:out_of_t, :3].copy()

				if remove_errors: # Remove errors from threat tracking
					tracking = cleanup(tracking, speed_th)
					s_tracking = cleanup(s_tracking, speed_th)
					t_tracking = cleanup(t_tracking, speed_th)
					n_tracking = cleanup(n_tracking, speed_th)

				out_of_ts.append(out_of_t)
				threat_trackings.append(tracking)
				s_threat_trackings.append(s_tracking); t_threat_trackings.append(t_tracking)
				n_threat_trackings.append(n_tracking)
				speeds_at_out_t.append(tracking[-1, -1])
				maze_ids.append(int(condition[-1]))

			trials["frame_out_of_t"] = out_of_ts
			trials["threat_tracking"] = threat_trackings
			trials["snout_threat_tracking"], trials["tail_threat_tracking"] = s_threat_trackings, t_threat_trackings
			trials['neck_threat_tracking'] = n_threat_trackings
			trials["speed_at_out_t"] = speeds_at_out_t
			trials["maze_id"] = maze_ids

			self.trials[condition] = trials

		self.all_trials_tplatf = pd.concat(list(self.trials.values()))

