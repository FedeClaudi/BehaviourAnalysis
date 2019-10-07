import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project
from Utilities.imports import *
from Processing.plot.plot_distributions import plot_fitted_curve, dist_plot
import statsmodels.api as sm

from scipy.signal import find_peaks, resample

class timedAnalysis:
	def __init__(self):
		pass

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

	def timed_plots_for_upgrade(self):
		def plot_fitparams(ax, res):
		   ax.text(0.95, 0.8, '$slope:{}\\ -\\ p:{}\\ -\\ r^2:{}$'.format(round(res.params[1], 5), round(res.pvalues[1], 8), round(res.rsquared, 3)), 
						color="k", fontsize=32, transform=ax.transAxes, **text_axaligned)
			
		grouped_modes, grouped_means, grouped_params, _, _ = self.bayes_by_condition_analytical(mode="grouped", plot=False) 
		rtdf = self.inspect_rt_metric(load=True, plot=False)

		# crate figure
		f, axarr = create_figure(subplots=True, nrows=3, sharex=True)

		# ? Plot timed PR
		ax = axarr[0]
		window_size = 20*60  # in seconds
		magification_factor = 100
		n_steps = 80

		# loop over experiments
		colors_helper = MplColorHelper("Purples", 0, 5, inverse=True)
		colors = [colors_helper.get_rgb(i) for i in range(len(list(self.conditions.keys())))]
		for i, (condition, trials) in enumerate(self.conditions.items()):
			# Get escape arms by time
			trial_times = trials.stim_frame_session.values / trials.fps.values   # times are in seconds
			trial_outcomes = np.array([1 if "Right" in t.escape_arm else 0 for i,t in trials.iterrows()])

			trial_outcomes = trial_outcomes[np.argsort(trial_times)]
			trial_times = np.sort(trial_times)

			# Sweep over time and do windowed analytical bayes
			means, std = [], []

			times, means, stds = [], [], []
			# for t in np.linspace(np.min(trial_times), np.max(trial_times), n_steps):
			_times = np.arange(10*60, 80*60, window_size)
			for t in _times:   # loop over the whole session takin N s everytime

				in_window = np.where((trial_times > t-window_size/2) & (trial_times < t+window_size/2))
				if np.any(in_window):
					outcomes_in_window = trial_outcomes[in_window]

					if len(outcomes_in_window) < 5: continue # skip times when there are too few trials
					
					(a, b, fact), mean, var = self.simple_analytical_bayes(outcomes_in_window)
					means.append(mean)
					times.append(t)


			ax.errorbar(times, means, yerr=None, 
				fmt='o', markeredgecolor=black, markerfacecolor=colors[i], markersize=50, 
				ecolor=desaturate_color(colors[i], k=.7), elinewidth=12, label=condition,
				capthick=2, alpha=1, zorder=20, 
				color=[.2, .2, .2], ls="--", lw=10) 

		ax.set(ylim=[0, 1],  ylabel="$p(R)$", xticks=[x for x in _times], xticklabels=["${}$".format(np.int(x/60)) for x in _times], xlim=[0, 80*60])


		# ? Plot timed RT and escape speed

		all_speeds, all_rts, all_times, all_times2 = [], [], [], []
		for i, (condition, df) in enumerate(self.conditions.items()):
			times, times2, ones, zeros, speeds, rts, = [], [], [], [], [], [],
			# loop over trials
			for n, (_, trial) in enumerate(df.iterrows()):
				x = trial.stim_frame_session / trial.fps

				escape_speed = np.mean(line_smoother(trial.tracking_data[:, 2], window_size=51, order=5)) / trial.fps
				times.append(x)
				speeds.append(escape_speed)

				# Get reaction time
				rt = rtdf.loc[rtdf.trialid == trial.trial_id].rt_s.values
				if np.any(rt): 
					if not np.isnan(rt[0]):
						times2.append(x)
						rts.append(rt[0])

			all_speeds.extend(speeds)
			all_rts.extend(rts)
			all_times.extend(times)
			all_times2.extend(times2)

		# plot
		# OLS params are ordered: Intercept, slope
		axarr[1].scatter(all_times, all_speeds, s=25, color=[.2, .2, .2])
		_, p0, p1, res = linear_regression(all_times, all_speeds, robust=False)
		plot_fitparams(axarr[1], res)
		xmax = 90*60
		axarr[1].plot([np.min(all_times), xmax], [0*p1+p0, xmax*p1+p0], color=red)

		axarr[2].scatter(all_times2, all_rts, s=25, color=[.2, .2, .2])
		_, p0, p1, res = linear_regression(all_times2, all_rts, robust=False)
		plot_fitparams(axarr[2], res)
		axarr[2].plot([np.min(all_times2), xmax], [0*p1+p0, xmax*p1+p0], color=red)


		axarr[1].set(title="$\\textrm{Mean escape speed}$", ylabel="$(a.u.)$")
		axarr[2].set(title="$\\textrm{Reaction time}$", ylabel="$s$", xlabel="$min$")

		sns.despine(fig=f, offset=10, trim=False, left=False, right=True)
