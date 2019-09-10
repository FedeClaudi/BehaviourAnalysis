
import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project

from statistics import mode
import pymc3 as pm
from sklearn.metrics import mean_squared_error as MSE
from scipy.spatial.distance import squareform
# from lmfit import minimize, Parameters, Model

from Utilities.imports import *

from Processing.plot.plot_distributions import plot_fitted_curve, dist_plot
from Processing.analyse_experiments import ExperimentsAnalyser
from Processing.rt_analysis import rtAnalysis
from Processing.timed_analysis import timedAnalysis
from Processing.timeseries_analysis import TimeSeriesAnalysis


class PsychometricAnalyser(ExperimentsAnalyser, rtAnalysis, timedAnalysis, TimeSeriesAnalysis):
	maze_names = {"maze1":"asymmetric_long", "maze2":"asymmetric_mediumlong", "maze3":"asymmetric_mediumshort", "maze4":"symmetric"}

	# DT simulation params
	speed_mu, speed_sigma = 5, 2.5
	speed_noise = 0
	distance_noise = .1

	#  ! important param
	ratio = "georatio"  # ? Use  either georatio or ratio for estimating L/R length ratio

	def __init__(self, naive=None, lights=1, escapes=True, escapes_dur=True):
		ExperimentsAnalyser.__init__(self, naive=naive, lights=lights, escapes=escapes, escapes_dur=escapes_dur)
		# rtAnalysis.__init__(self)

		self.conditions = self.load_trials_from_pickle()
		self.maze_names_r = {v:k for k,v in self.maze_names.items()}

		self.get_paths_lengths()

	def get_paths_lengths(self, load=True):
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

			self.paths_lengths = pd.DataFrame.from_dict(summary)
			self.paths_lengths.to_pickle(os.path.join(self.metadata_folder, "path_lengths.pkl"))
		else:
			self.paths_lengths = pd.read_pickle(os.path.join(self.metadata_folder, "path_lengths.pkl"))

		geopaths = self.get_arms_lengths_with_agent(load=True)
		self.paths_lengths = pd.merge(self.paths_lengths, geopaths)
		short_arm_dist = self.paths_lengths.loc[self.paths_lengths.maze=="maze4"].distance.values[0]
		self.paths_lengths["georatio"] = [round(x / short_arm_dist, 4) for x in self.paths_lengths.distance.values]

		self.short_arm_len = self.paths_lengths.loc[self.paths_lengths.maze=="maze4"][self.ratio].values[0]
		self.long_arm_len = self.paths_lengths.loc[self.paths_lengths.maze=="maze1"][self.ratio].values[0]


	"""
		||||||||||||||||||||||||||||    GETTERS     |||||||||||||||||||||
	"""
	def get_hb_modes(self, trace=None):
		if trace is None:
			trace = self.bayes_by_condition(conditions=self.conditions, load=True, tracefile="psychometric_individual_bayes.pkl", plot=False)
		
		n_bins = 100
		bins = {k:[np.digitize(a, np.linspace(0, 1, n_bins)) for a in v.T] for k,v in trace.items()}
		modes = {k:[np.median(b)/n_bins for b in bins] for k,bins in bins.items()}
		stds = {k:[np.std(m) for m in v.T] for k,v in trace.items()}
		means = {k:[np.mean(m) for m in v.T] for k,v in trace.items()}

		return modes, means, stds, trace

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
		print(result.params["sigma"].value)
		a = 1


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
		||||||||||||||||||||||||||||    BAYES     |||||||||||||||||||||
	"""
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


if __name__ == "__main__":
	pa = PsychometricAnalyser()

	# pa.plot_pr_by_condition_detailed()
	# pa.model_summary() 
	# pa.plot_escape_duration_by_arm()
	# pa.plot_hierarchical_bayes_effect()

	# pa.inspect_rt_metric(load=False)

	# pa.plot_effect_of_time(xaxis_istime=False, robust=False) # ? not useful
	# pa.plot_effect_of_time(xaxis_istime=True, robust=False)
# 
	pa.timed_plots_for_upgrade()

	# pa.closer_look_at_hb()

	# pa.test_hierarchical_bayes_v2()
	# pa.test()
	# pa.inspect_hbv2()

	print(pa.paths_lengths)

	plt.show()



