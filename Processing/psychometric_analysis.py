# %%
import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project

from statistics import mode
import pymc3 as pm
from sklearn.metrics import mean_squared_error as MSE
from scipy.spatial.distance import squareform
from lmfit import minimize, Parameters, Model

from Utilities.imports import *

from Processing.plot.plot_distributions import plot_fitted_curve, dist_plot
from Processing.analyse_experiments import ExperimentsAnalyser
from Processing.rt_analysis import rtAnalysis
from Processing.timed_analysis import timedAnalysis
from Processing.timeseries_analysis import TimeSeriesAnalysis

# %%
# Define class

class PsychometricAnalyser(ExperimentsAnalyser, rtAnalysis, timedAnalysis, TimeSeriesAnalysis):
	maze_names = {"maze1":"asymmetric_long", "maze2":"asymmetric_mediumlong", "maze3":"asymmetric_mediumshort", "maze4":"symmetric"}

	# DT simulation params
	speed_mu, speed_sigma = 5, 2.5
	speed_noise = 0
	distance_noise = .1

	def __init__(self):
		ExperimentsAnalyser.__init__(self)
		rtAnalysis.__init__(self)

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

		self.short_arm_len = self.paths_lengths.loc[self.paths_lengths.maze=="maze4"].georatio.values[0]
		self.long_arm_len = self.paths_lengths.loc[self.paths_lengths.maze=="maze1"].georatio.values[0]


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
		distances = {a.maze:stats.norm(loc=a.georatio, scale=dnoise) for i,a in self.paths_lengths.iterrows()}
		
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
			x = self.paths_lengths.loc[self.paths_lengths.maze == condition].georatio.values
			y = means[condition]

			# ? plot HB PR with errorbars
			ax.errorbar(x, np.mean(y), yerr=np.std(y), 
						fmt='o', markeredgecolor=self.colors[i+1], markerfacecolor=self.colors[i+1], markersize=15, 
						ecolor=desaturate_color(self.colors[i+1], k=.7), elinewidth=3, 
						capthick=2, alpha=1, zorder=0)             

		# ? Fit model with lmfit
		# def residual(params, ytrue):
		#     self.distance_noise = params["sigma"].value

		#     analytical_pr = self.simulate_trials_analytical()
		#     # return analytical_pr
		#     return np.sum((np.array(list(analytical_pr.values())) - ytrue)**2)

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
		x = self.paths_lengths.georatio.values

		result = model.fit(ytrue, distances=x, params=params)
		print(result.params["sigma"].value)
		a = 1

		# params = Parameters()
		# params.add("sigma", min=1.e-10, max=.5, brute_step=.01)
		# 
		# out = minimize(residual, params, args=(ytrue,)) #, method="brute")
		# print(out.params["sigma"])
		# a = 1
	
		# # ? Fit the model with a range of params and plot the results
		# fitted = {k:[]  for k in self.conditions.keys()}
		# mserr = []
		# minsigma, maxsigma = 0.05, .2
		# sigma_range = np.linspace(minsigma, maxsigma, 100)
		# ytrue = [np.mean(m) for m in means.values()]
		# for sigma in sigma_range:
		#     self.distance_noise = sigma
		#     analytical_pr = self.simulate_trials_analytical()
		#     simulation = self.simulate_trials_analytical()
		#     {fitted[k].append(pr) for k,pr in simulation.items()}
		#     mserr.append(MSE(ytrue, list(simulation.values())))
   
		# # ? Plot mean square error
		# for s in sigma_range[::5]:
		#     vline_to_curve(mseax, s, sigma_range, mserr, color=desaturate_color(teal), lw=2)
		# mseax.plot(sigma_range, mserr, color=teal, lw=4)
		# mseax.axhline(0, color=white, lw=4)

		# ? Plot best fit
		# best_sigma = sigma_range[np.argmin(mserr)]
		best_sigma = result.params["sigma"].value
		self.distance_noise = best_sigma
		# lowest_err = mserr[np.argmin(mserr)]
		# vline_to_curve(mseax, best_sigma, sigma_range, mserr, color=white, lw=6)
		# hline_to_curve(mseax, lowest_err, sigma_range, mserr, color=white, lw=6)

		analytical_pr = self.simulate_trials_analytical()
		pomp = plot_fitted_curve(sigmoid, self.paths_lengths.georatio.values, np.hstack(list(analytical_pr.values())), ax, xrange=xrange, 
			scatter_kwargs={"alpha":0}, 
			line_kwargs={"color":white, "alpha":1, "lw":6, "label":"model pR - $\sigma : {}$".format(round(best_sigma, 2))})


		# Fix plotting
		ortholines(ax, [1, 0,], [1, .5])
		ortholines(ax, [0, 0,], [1, 0], ls=":", lw=1, alpha=.3)
		ax.set(title="best fit logistic regression", ylim=[-0.01, 1.05], ylabel="p(R)", xlabel="Left path length (a.u.)",
				 xticks = self.paths_lengths.georatio.values, xticklabels = self.conditions.keys())
		# mseax.set(title="Fit error", ylabel="MSE", xlabel="$\sigma$", xlim=[minsigma, maxsigma], ylim=[0, max(mserr)])
		make_legend(ax)
		

	"""
		||||||||||||||||||||||||||||    BAYES     |||||||||||||||||||||
	"""

	def sigmoid_bayes(self, plot=True, load=False, robust=False):
		tracename = os.path.join(self.metadata_folder, "robust_sigmoid_bayes.pkl")
		if not load:
			# Get data
			allhits, ntrials, p_r, n_mice = self.get_binary_trials_per_condition(self.conditions)
			
			# Clean data and plot scatterplot
			if plot: f, ax = plt.subplots(figsize=large_square_fig)
			x_data, y_data = [], []
			for i, (condition, hits) in enumerate(allhits.items()):
				failures = [ntrials[condition][ii]-hits[ii] for ii in np.arange(n_mice[condition])]            
				x = self.paths_lengths.loc[self.paths_lengths.maze == condition].georatio.values[0]

				xxh, xxf = [x for h in hits for _ in np.arange(h)],   [x for f in failures for _ in np.arange(f)]
				yyh, yyf = [1 for h in hits for _ in np.arange(h)],   [0 for f in failures for _ in np.arange(f)]

				x_data += xxh + xxf
				y_data += yyh + yyf

			if plot:
				ax.scatter(x_data, [y + np.random.normal(0, 0.07, size=1) for y in y_data], color=white, s=250, alpha=.3)
				ax.axvline(1, color=grey, alpha=.8, ls="--", lw=3)
				ax.axhline(.5, color=grey, alpha=.8, ls="--", lw=3)
				ax.axhline(1, color=grey, alpha=.5, ls=":", lw=1)
				ax.axhline(0, color=grey, alpha=.5, ls=":", lw=1)

			# Get bayesian logistic fit + plot
			xp = np.linspace(np.min(x_data)-.2, np.max(x_data)  +.2, 100)
			if not robust:
				trace = self.bayesian_logistic_regression(x_data, y_data) # ? naive
			else:
				trace = self.robust_bayesian_logistic_regression(x_data, y_data) # ? robust

			b0, b0_std = np.mean(trace.get_values("beta0")), np.std(trace.get_values("beta0"))
			b1, b1_std = np.mean(trace.get_values("beta1")), np.std(trace.get_values("beta1"))
			if plot:
				ax.plot(xp, logistic(xp, b0, b1), color=red, lw=3)
				ax.fill_between(xp, logistic(xp, b0-b0_std, b1-b1_std), logistic(xp, b0+b0_std, b1+b1_std),  color=red, alpha=.15)
		
				ax.set(title="Logistic regression", yticks=[0, 1], yticklabels=["left", "right"], ylabel="escape arm", xlabel="L/R length ratio",
							xticks=self.paths_lengths.georatio.values, xticklabels=self.paths_lengths.georatio.values)

			df = pd.DataFrame.from_dict(dict(b0=trace.get_values("beta0"), b1=trace.get_values("beta1")))
			df.to_pickle(tracename)
		else:
			df = pd.read_pickle(tracename)
		return df

	def closer_look_at_hb(self):
		# Get paths length ratios and p(R) by condition
		hits, ntrials, p_r, n_mice, trials = self.get_binary_trials_per_condition(self.conditions)
		modes, means, stds, traces = self.get_hb_modes()
		f, axarr = create_figure(subplots=False, nrows=4, sharex=True)

		for i, (condition, trace) in enumerate(traces.items()):
			sort_idx = np.argsort(means[condition])
			nmice = len(sort_idx)
			above_chance, below_chance = 0, 0
			for mn, id in enumerate(sort_idx):
				tr = trace[:, id]
				# Plot raw PR
				axarr[i].scatter(mn+.1, p_r[condition][id], color=teal, s=25, alpha=.8)

				# Plot KDE of posterior
				kde = fit_kde(random.choices(tr,k=5000), bw=.025)
				plot_kde(axarr[i], kde, z=mn, vertical=True, normto=.75, color=self.colors[i+1], lw=.5)

				# plot 95th percentile_range of posterior's means
				percrange = percentile_range(tr)
				axarr[i].scatter(mn, percrange.mean, color=self.colors[i+1], s=25)
				axarr[i].plot([mn, mn], [percrange.low, percrange.high], color=grey, lw=2, alpha=.4)
				if percrange.low > .5: above_chance += 1
				elif percrange.high < .5: below_chance += 1

			axarr[i].text(0.95, 0.1, '{}% above .5 - {}% below .5'.format(round(above_chance/nmice*100, 2), round(below_chance/nmice*100, 2)), color=grey, fontsize=15, transform=axarr[i].transAxes, **text_axaligned)

			axarr[i].set(ylim=[0, 1], ylabel=condition)
			axarr[i].axhline(.5, **grey_dotted_line)
		
		axarr[0].set(title="HB posteriors")
		axarr[-1].set(xlabel="mouse id")

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


	def test(self):
		# Add the KDE's for each mouse in an expeirment and plot
		f, ax = plt.subplots()
		trace = self.load_trace(os.path.join(self.metadata_folder, "test_hb_trace.pkl"))
		data = self.get_hits_ntrials_maze_dataframe()

		comulative_trace = {k:np.zeros(2000) for k,v in self.conditions.items()}
		n_mices = {k:0 for k,v in self.conditions.items()}
		for column in trace:	
			if not "beta_theta" in column: continue
			n = int(column.split("_")[-1])
			mouse = data.iloc[n]
			
			comulative_trace["maze{}".format(mouse.maze + 1)] += trace[column]
			n_mices["maze{}".format(mouse.maze + 1)] += 1

		for i, (k, v) in enumerate(comulative_trace.items()):
			kde = fit_kde(v/n_mices[k],   bw=.025)
			plot_kde(ax, kde, invert=False, color=self.colors[i+1], z=0, alpha=.2, label=k)

		make_legend(ax)
		ax.set(ylabel="probability", xlabel="p(R)", xlim=[0, 1])

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
		grouped_modes, grouped_means, grouped_params = self.bayes_by_condition_analytical(mode="grouped", plot=False) 

		 # Plot each individual's pR and the group mean as a factor of L/R length ratio
		if ax is None: 
			f, ax = create_figure(subplots=False)
		else: f = None
			
		lr_ratios_mean_pr = {"grouped":[], "individuals_x":[], "individuals_y":[], "individuals_y_sigma":[]}
		for i, (condition, pr) in enumerate(p_r.items()):
			x = self.paths_lengths.loc[self.paths_lengths.maze == condition].georatio.values

			y = means[condition]
			# ? plot HB PR with errorbars
			ax.errorbar(np.random.normal(x, 0.005, size=len(y)), y, yerr=stds[condition], 
						fmt='o', markeredgecolor=desaturate_color(white, k=.6), markerfacecolor=desaturate_color(white, k=.6), markersize=10, 
						ecolor=desaturate_color(white, k=.2), elinewidth=3, 
						capthick=2, alpha=.6, zorder=0)
			ax.errorbar(x, np.mean(y), yerr=np.std(y), 
						fmt='o', markeredgecolor=self.colors[i+1], markerfacecolor=self.colors[i+1], markersize=15, 
						ecolor=desaturate_color(self.colors[i+1], k=.7), elinewidth=3, label=condition,
						capthick=2, alpha=1, zorder=0)             
			vline_to_point(ax, x, np.mean(y), color=desaturate_color(self.colors[i+1], k=.7), lw=4, ls="--", alpha=.2)


			if condition not in exclude_experiments:# ? use the data for curves fitting
				k = .4
				lr_ratios_mean_pr["grouped"].append((x[0], np.mean(pr), np.std(pr)))  
				lr_ratios_mean_pr["individuals_x"].append([x[0] for _ in np.arange(len(y))])
				lr_ratios_mean_pr["individuals_y"].append(y)
				lr_ratios_mean_pr["individuals_y_sigma"].append(stds[condition])
			else: 
				k = .1
				del grouped_modes[condition], grouped_means[condition]

		# Fix plotting
		ortholines(ax, [1, 0,], [1, .5])
		ax.set(ylim=[0, 1], ylabel="p(R)", title="p(R) per mouse per maze", xlabel="Left path length (a.u.)",
				 xticks = self.paths_lengths.georatio.values, xticklabels = self.conditions.keys())
		make_legend(ax)

		return lr_ratios_mean_pr, grouped_modes, grouped_means, modes, means, stds, f, ax, xp, xrange, grouped_params

	def plot_pr_by_condition_detailed(self):
		for bw in [0.01, 0.02, 0.03, 0.05, 0.1]:
			f, axarr = create_figure(subplots=True, ncols=5, sharey=False)
			# plot normal pR
			lr_ratios_mean_pr, grouped_modes, grouped_means, modes, means, stds, _, ax, xp, xrange, grouped_params = self.pr_by_condition(ax=axarr[0])

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
				plot_shaded_withline(axarr[1],xx, yy, color=self.colors[i+1], lw=3, label=maze, zorder=10 )

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
		lr_ratios_mean_pr, grouped_modes, grouped_means, modes, means, stds, f, ax, xp, xrange, _ = self.pr_by_condition(exclude_experiments=exclude_experiments, ax=ax)

		# Plot simulation results   + plotted sigmoid
		# ? logistic regression on analytical simulation

		fitted = []
		for i, (nn, col) in enumerate(zip(np.linspace(.05, .2, 2), get_n_colors(10))):
			self.distance_noise = nn
			analytical_pr = self.simulate_trials_analytical()
			pomp = plot_fitted_curve(sigmoid, [m[0] for m in lr_ratios_mean_pr["grouped"]], np.hstack(list(analytical_pr.values())), ax, xrange=xrange, 
						scatter_kwargs={"alpha":0}, 
						line_kwargs={"color":desaturate_color(teal), "alpha":0, "lw":4})
			fitted.append(pomp)
		ax.fill_between(xp, sigmoid(xp, *fitted[0]), sigmoid(xp, *fitted[1]),  color=lightblue, alpha=.15, label="model pR - $\sigma : {}-{}$".format(.05, .2))

		# ? Fit sigmoid to median pR of raw data  
		plot_fitted_curve(sigmoid, [m[0] for m in lr_ratios_mean_pr["grouped"]], [m[1] for m in lr_ratios_mean_pr["grouped"]], ax, xrange=xrange, 
								fit_kwargs={"sigma":[m[2] for m in lr_ratios_mean_pr["grouped"]]},
								scatter_kwargs={"alpha":0}, 
								line_kwargs={"color":pink, "alpha":.8, "lw":4, "label":"mean raw p(R)"})

		# ? Fit logistic regression to mean p(R)+std(p(R))
		pomp = plot_fitted_curve(sigmoid, [m[0] for m in lr_ratios_mean_pr["grouped"]], [np.mean(n) for n in lr_ratios_mean_pr["individuals_y"]], ax, xrange=xrange, 
			fit_kwargs={"sigma":[np.std(n) for n in lr_ratios_mean_pr["individuals_y"]]},
			scatter_kwargs={"alpha":0}, 
			line_kwargs={"color":teal, "alpha":.5, "lw":4, "label":"mean hb p(R)"})

		# ? Plot fitted sigmoid
		hbfit = plot_fitted_curve(sigmoid, np.hstack(lr_ratios_mean_pr["individuals_x"]), np.hstack(lr_ratios_mean_pr["individuals_y"]), ax, xrange=xrange,  # ? ind. sigmoid
								fit_kwargs={"sigma": np.hstack(lr_ratios_mean_pr["individuals_y_sigma"]), }, 
								scatter_kwargs={"alpha":0}, 
								line_kwargs={"color":grey, "alpha":.8, "lw":4, "label":"individuals hb p(R)s"})

		# Fix plotting
		ortholines(ax, [1, 0,], [1, .5])
		ortholines(ax, [0, 0,], [1, 0], ls=":", lw=1, alpha=.3)
		ax.set(ylim=[-0.01, 1.05], ylabel="p(R)", title="p(R) per mouse per maze", xlabel="Left path length (a.u.)",
				 xticks = self.paths_lengths.georatio.values, xticklabels = self.conditions.keys())
		make_legend(ax)

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

	def plot_utility_function(self):
		def get_utility_matrix():
			n_cols = 500
			x = np.linspace(0, 2, n_cols)
			y = np.linspace(0, 2, n_cols)

			u = np.reshape([2 - (xx-yy) for xx in x for yy in y], (n_cols, n_cols)).T
			up_tr = np.triu(u) 
			up_tr[np.tril_indices(n_cols)] = np.nan
			return np.rot90(up_tr.T, 1)


		print(self.paths_lengths)
		
		f, axarr = create_figure(subplots=True, ncols=2)
		ax, ax2 = axarr[0], axarr[1]

		# Get background image and show
		util = get_utility_matrix()
		ax.imshow(util, extent=[.8, 1.6, .8, 1.6], alpha=.9)

		# Plt arms combinations
		distances = {}
		for i, row in self.paths_lengths.iterrows():
			maze_dist = calc_distance_between_point_and_line([[.8, .8], [1.6, 1.6]], [self.short_arm_len, row.georatio])
			ax.scatter(self.short_arm_len, row.georatio, color=self.colors[i+1], label=row.maze + " - {}".format(round(maze_dist, 4)), **big_dot)
			distances[row.maze] = maze_dist
			c = row.georatio - self.short_arm_len
			ax.plot([0.5, 1.75], [.5+c, 1.75+c], color=self.colors[i+1], lw=3, ls="--", alpha=.25)

			if i < 3: 
				maze_dist2 = calc_distance_between_point_and_line([[.8, .8], [1.6, 1.6]], [row.georatio, self.long_arm_len])
				ax.scatter(row.georatio, self.long_arm_len, color=random.choice(get_n_colors(10)), label=round(maze_dist2, 4),  **big_dot)

		ax.plot([0.5, 1.75], [.5, 1.75], **grey_line)

		make_legend(ax)
		ax.set(title="utility function", xlabel="Right arm length (a.u.)", ylabel="Left arm length (a.u.)",
				xlim=[.8, 1.6], ylim=[.8, 1.6])

		# Plot p(R) vs distance from line
		hits, ntrials, p_r, n_mice = self.get_binary_trials_per_condition(self.conditions)
		modes, means, stds = self.get_hb_modes()
		for_fitting = {"x":[], "y":[], "e":[]}
		for i, (maze, pr) in enumerate(p_r.items()):
			x, y, e = np.random.normal(distances[maze], 0.005, size=len(means[maze])), means[maze], stds[maze]
			ax2.errorbar(x, y, yerr=e, 
						fmt='o', markeredgecolor=self.colors[i+1], markerfacecolor=self.colors[i+1], markersize=10, 
						ecolor=desaturate_color(self.colors[i+1], k=.4), elinewidth=3,  capthick=2, alpha=.8, zorder=0)
			for_fitting["x"].append(x)
			for_fitting["y"].append(y)
			for_fitting["e"].append(e)


		# ? Fit logistic regression to mean p(R)+std(p(R))
		# pomp = plot_fitted_curve(sigmoid, [np.mean(a) for a in for_fitting["x"]], [np.mean(a) for a in for_fitting["y"]], ax2, xrange=[-.01, .45], 
		#     fit_kwargs={"sigma":[np.mean(a) for a in for_fitting["e"]]},
		#     scatter_kwargs={"alpha":1, "color":lightblue}, 
		#     line_kwargs={"color":lightblue, "alpha":1, "lw":4}) 

		pomp = plot_fitted_curve(sigmoid, np.hstack(for_fitting["x"]), np.hstack(for_fitting["y"]), ax2, xrange=[-.01, .45], 
			fit_kwargs={"sigma":np.hstack(for_fitting["e"])},
			scatter_kwargs={"alpha":0}, 
			line_kwargs={"color":white, "alpha":1, "lw":4}) 

		ortholines(ax2, [0, 0.5,], [.5, 0], ls="--", lw=3, alpha=.3)
		ax2.set(title="p(R) vs distance from unity line", xlabel="distance", ylabel="p(R)")

		plt.show()
		a =1 

	





if __name__ == "__main__":
	pa = PsychometricAnalyser()

	# pa.plot_pr_by_condition_detailed ()
	# pa.model_summary()
	# pa.plot_hierarchical_bayes_effect()

	# pa.inspect_rt_metric(load=False)

	# pa.plot_effect_of_time(xaxis_istime=False)
	# pa.plot_effect_of_time(xaxis_istime=True, robust=True)

	# pa.timed_pr()

	# pa.closer_look_at_hb()

	pa.test_hierarchical_bayes_v2()
	pa.test()
	# pa.inspect_hbv2()

	print(pa.paths_lengths)

	plt.show()



#%%
