import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project

from Utilities.imports import *


import pickle
# import pymc3 as pm
import statsmodels.api as sm
from scipy.signal import find_peaks, resample, medfilt

from Modelling.bayes import Bayes
from Modelling.maze_solvers.gradient_agent import GradientAgent
from Modelling.maze_solvers.environment import Environment
from Analysis.Behaviour.utils.trials_data_loader import TrialsLoader
from Analysis.Behaviour.utils.path_lengths import PathLengthsEstimator
from Analysis.Behaviour.plotting.plot_trials_tracking import TrialsPlotter
from Analysis.Behaviour.utils.plots_by_condition import PlotsByCondition

"""[This class facilitates the loading of experiments trials data + collects a number of methods for the analysis. ]
"""


class ExperimentsAnalyser(Bayes, Environment, TrialsLoader, PathLengthsEstimator, TrialsPlotter, PlotsByCondition):
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


	def __init__(self, naive=None, lights=None, escapes_dur=None, shelter=True, 
			agent_params=None, **kwargs):
		# Initiate some parent classes
		Bayes.__init__(self) # Get functions from bayesian modelling class
		PathLengthsEstimator.__init__(self)
		TrialsPlotter.__init__(self)
		PlotsByCondition.__init__(self)
		
		# store params
		self.naive, self.lights, self.escapes_dur, self.shelter = naive, lights, escapes_dur, shelter

		# Get trials data
		TrialsLoader.__init__(self, naive=self.naive, lights=self.lights, 
						escapes_dur=self.escapes_dur, shelter=self.shelter, **kwargs)

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
	def get_binary_trials_per_condition(self, conditions, ignore_center=True):
		# ? conditions should be a dict whose keys should be a list of strings with the names of the different conditions to be modelled
		# ? the values of conditions should be a a list of dataframes, each specifying the trials for one condition (e.g. maze design) and the session they belong to

		# Parse data
		# Get trials
		trials = {c:[] for c in conditions.keys()}
		for condition, df in conditions.items():
			sessions = sorted(set(df.uid.values))
			for sess in sessions:
				if "center" in  df.loc[df.uid==sess].escape_arm.values:
					if not ignore_center:
						raise NotImplementedError
				else:
					df = df.loc[df.escape_arm != "center"]
				trials[condition].append([1 if "right" in arm.lower() else 0 for arm in df.loc[df.uid==sess].escape_arm.values])

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
	||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
						 BAYES
	||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
	"""
	def bayes_by_condition_analytical(self, load=True, mode="grouped", plot=True):
		results = {"condition":[], "mean":[], "median":[], "sigmasquared":[], "prange":[]}
		hits, ntrials, p_r, n_mice, trials = self.get_binary_trials_per_condition(self.conditions)

		for (cond, h), n in zip(hits.items(), ntrials.values()):
			res = self.grouped_bayes_analytical(n, h)
			results['condition'].append(cond)
			results['mean'].append(res[0])
			results['median'].append(res[1])
			results['sigmasquared'].append(res[2])
			results['prange'].append(res[3])

		return pd.DataFrame(results)


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
	
