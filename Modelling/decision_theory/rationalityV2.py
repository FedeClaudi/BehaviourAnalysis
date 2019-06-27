# %%
import sys
sys.path.append("./")
from Utilities.imports import *
from Modelling.glm.glm_data_loader import GLMdata
from Modelling.maze_solvers.gradient_agent import GradientAgent as GeoAgent
from Processing.trials_analysis.all_trials_loader import Trials

from scipy import stats
from scipy.optimize import curve_fit

data = GLMdata

%matplotlib inline


# %%
# Define class
class Analyzer(GLMdata):
	arms_names = ["centre", "rightmedium", "leftfar"]
	experiments_names = ["threearms", "symmetric", "asymmetric"]
	colors = {"centre":"m", "rightmedium":"g", "leftfar":"r", "leftmedium":"g"}
	colors2 = {"centre":"m", "right_medium":"g", "left_far":"r", "left_medium":"g"}

	def __init__(self, experiment):
		GLMdata.__init__(self, load_trials_from_file=True)

		# Params for simulations
		self.pathlength_noise_factor = 2
		self.speed_noise_param = 1

		# Process trials data
		self.augment_trials()
		self.remove_outliers()

		# keep a copy fo all trials
		self.all_trials = self.trials.copy()

		# ARms params
		self.keep_trials_experiment("threearms") # necessary to correctly estimate path length
		self.params = self.load_maze_params()
		self.get_arms_params_with_geoagent()

		# Keep only trial for the experiment
		self.keep_trials_experiment(experiment)

		# other
		self.get_speed_distributions()

	def keep_trials_experiment(self, experiment):
		self.experiment_name = experiment
		self.trials = self.all_trials.loc[self.all_trials["experiment_{}".format(experiment)] == 1]
		self.get_trials_by_arm()
		self.get_speed_distributions()

	def augment_trials(self):
		# Calculate stuff like escape duration...
		trials = self.trials
		trials["escape_duration_frames"] = [t.shape[0] for t in trials.tracking_data_trial.values]
		trials["escape_duration_s"] = np.array(trials.escape_duration_frames.values / trials.fps.values, np.float32)
		trials["log_escape_duration_s"] = np.log(np.array(trials.escape_duration_frames.values / trials.fps.values, np.float32))
		trials["escape_distance"] = [np.sum(calc_distance_between_points_in_a_vector_2d(t[:, :2])) for t in trials.tracking_data_trial.values]

	def remove_outliers(self):
		# remove trials whose escape distance is too high for them to be real escapes
		self.trials = self.trials.loc[self.trials.escape_distance < 1000]

	def get_trials_by_arm(self, exp="asymmetric"):
		trials = self.trials
		if exp != "symmetric": self.left = trials.loc[trials.escape_arm == "Left_Far"]
		else: self.left = trials.loc[trials.escape_arm == "Left_Medium"]
		self.right = trials.loc[trials.escape_arm == "Right_Medium"]
		self.centre = trials.loc[trials.escape_arm == "Centre"]

	def get_speed_distributions(self):
		expl_speeds = np.hstack([t[:, 2] for t in self.trials.tracking_data_exploration])
		trials_speeds = np.hstack([t[:, 2] for t in self.trials.tracking_data_trial])
		speeds = np.hstack([expl_speeds, trials_speeds])

		self.exploration_speeds, self.trials_speeds, self.all_speeds = expl_speeds, trials_speeds, speeds

	def get_speed_modelled(self):
		self.speed_modelled = stats.norm(loc = np.mean(self.trials.mean_escape), scale=np.std(self.trials.mean_escape))

	def plot_speeds(self):
		self.get_speed_modelled()
		x = np.linspace(self.speed_modelled.ppf(.01), self.speed_modelled.ppf(.99), 100)

		f, ax = plt.subplots()
		ax.hist(self.trials.mean_speed, density=True, color="w", alpha=.3, label="mean trials speed")
		ax.plot(x, self.speed_modelled.pdf(x), color="orange", label="escape speed distribution")

		ax.set(xlim=[0, 8], facecolor=[.2, .2, .2], xlabel="speed (a.u.)", ylabel="p")
		ax.legend()

	def get_arms_params_with_geoagent(self):
		# Get arms params with geo agent
		if sys.platform == "darwin":
			folder = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/analysis_metadata/maze_solvers/good_single_arms"
		else:
			folder = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\maze_solvers\\good_single_arms"
		arms = [os.path.join(folder, a) for a in os.listdir(folder) if "jpg" in a]

		agent = GeoAgent(grid_size=500, start_location=[255, 125], goal_location=[255, 345])
		arms_data = dict(name=[], n_steps=[], distance=[], torosity=[], idistance_norm=[],
						max_distance=[], ideltateta_norm=[], iTheta=[])
		for arm in arms:
			# ? get maze, geodesic and walk
			agent.maze, agent.free_states = agent.get_maze_from_image(model_path=arm)
			agent.geodesic_distance = agent.geodist(agent.maze, agent.goal_location)

			walk = agent.walk()

			# ? evalueate walk
			shelter_distance = calc_distance_from_shelter(np.vstack(walk), agent.goal_location)
			arms_data["max_distance"].append(np.max(shelter_distance))
			arms_data["idistance_norm"].append(np.sum(shelter_distance)/len(walk))
			arms_data["name"].append(os.path.split(arm)[1].split(".")[0])
			arms_data["n_steps"].append(len(walk))
			arms_data["distance"].append(round(np.sum(calc_distance_between_points_in_a_vector_2d(np.array(walk)))))
			arms_data["iTheta"].append(np.sum(np.abs(calc_ang_velocity(calc_angle_between_points_of_vector(np.array(walk))[1:]))))

			threat_shelter_dist = calc_distance_between_points_2d(agent.start_location, agent.goal_location)
			arms_data["torosity"].append(np.sum(calc_distance_between_points_in_a_vector_2d(np.array(walk))) / threat_shelter_dist)

			# ? calculate normalised integral of delta theta (angle to shelter - dirction of motion)
			# angle to shelter at all points
			walk = np.vstack(walk)
			theta_shelter = []
			for p in walk:
				theta_shelter.append(angle_between_points_2d_clockwise(p, agent.goal_location))
			theta_shelter = np.array(theta_shelter)

			# calc direction of motion at all points during walk
			theta_walk = []
			for i in range(len(walk)):
				if i == 0: theta_walk.append(0)
				else: theta_walk.append(angle_between_points_2d_clockwise(walk[i-1], walk[i]))
			theta_walk = np.array(theta_walk)

			# integrate and normalise
			delta_theta = np.sum(np.abs(theta_shelter-theta_walk))/arms_data["n_steps"][-1] # <---
			arms_data["ideltateta_norm"].append(delta_theta)

		self.arms_params = pd.DataFrame(arms_data)

		# Correct arm distance
		mean_arm_distances = {"leftfar": np.mean(self.left.escape_distance),
								"rightmedium": np.mean(self.right.escape_distance),
								"centre": np.mean(self.centre.escape_distance)}
		conv_factors = {arm: d / self.arms_params.loc[self.arms_params["name"] == arm].distance.values[0] for arm,d in mean_arm_distances.items()}

		mean_right_dist = np.mean(self.right.escape_distance)
		conv_fact = mean_right_dist / self.arms_params.loc[self.arms_params["name"] == "rightmedium"].distance.values[0]
		self.arms_params["distance"] = [d*conv_factors[n] for d,n in zip(self.arms_params["distance"], self.arms_params["name"])]

	def get_pathlength_estimates(self, plot=False):
		moments = {arm["name"]:(arm.distance, math.sqrt(arm.distance)*self.pathlength_noise_factor) for i,arm in self.arms_params.iterrows()}
		distances = {arm["name"]:stats.norm(loc=arm.distance, scale=math.sqrt(arm.distance)*self.pathlength_noise_factor) for i,arm in self.arms_params.iterrows()}

		if plot:
			f, ax = plt.subplots()
			for a, d in distances.items():
				x = np.linspace(d.ppf(0.01), d.ppf(0.99), 100)
				ax.plot(x, d.pdf(x), color=self.colors[a], label=a, lw=3)
			ax.set(facecolor=[.2, .2, .2], xlabel="length (a.u.)", ylabel="p")
			ax.legend()
		
		return moments, distances

	def get_pathlength_from_rawdata(self):
		# For all trials get the mean_speed * duration -> estimate of path length
		distances = {arm:[] for arm in self.arms_names}
		distances_by_exp = {exp:{arm:[] for arm in self.arms_names} for exp in self.experiments_names}

		curr_exp = self.experiment_name
		for exp in self.experiments_names:
			self.keep_trials_experiment(exp)

			if exp == "threearms":
				trials = {"leftfar":self.left, "rightmedium":self.right, "centre":self.centre}
			elif exp == "asymmetric":
				trials = {"leftfar":self.left, "rightmedium":self.right}
			else:
				trials = {"rightmedium":self.trials}


			raw_dists = {n:t.mean_speed.values*t.escape_duration_s.values*30 for n,t in trials.items()} # ? *30 because of framerate
			distances_by_exp[exp] = raw_dists.copy()

			for arm in raw_dists.keys():
				distances[arm].extend(raw_dists[arm])

		moments = {n: (np.mean(d), np.std(d)) for n,d in distances.items()}
		model_dists = {n: stats.norm(loc=d[0], scale=d[1]) for n,d in moments.items()}

		return distances, distances_by_exp, model_dists, moments

	def plot_pr_rawdata(self):
		trial_data = Trials(exp_mode=0)
		trial_data.plot_parm_experiment()

	def plot_pr_simulations(self):
		f,axarr = plt.subplots(figsize=(10, 8), ncols=3, sharey=True)
		a.pathlength_noise_factor = 10
		a.speed_noise_param =  1.75

		a.get_pathlength_estimates(plot=True)

		for ax, exp in zip(axarr, a.experiments_names):
			a.keep_trials_experiment(exp)
			a.simulate_trials(ax=ax, n_trials=10000)

	def plot_duration_over_speed_raw(self, ax=None, original_colors=False):
		if ax is None:
			make_legend = True
			f, ax = plt.subplots(figsize=(8, 8))
		else:
			make_legend = False
			
		if not original_colors:
			colors = {a:"k" for a in self.arms_names}
		else:
			colors = self.colors

		ax.scatter(self.left.mean_speed.values, self.left.escape_duration_s.values, marker="v", color=colors["leftfar"], label="LEFT")
		ax.scatter(self.right.mean_speed, self.right.escape_duration_s, marker="o", color=colors["rightmedium"], label="RIGHT")
		ax.scatter(self.centre.mean_speed, self.centre.escape_duration_s, marker="*", color=colors["centre"], label="CENTRE")

		if make_legend:
			ax.legend()
			ax.set(facecolor=[.2, .2, .2], xlim=[0, 8], ylim=[0, 10], xlabel="speed (a.u.)", ylabel="duration (s)")

		return ax

	def plot_duration_over_speed_modelled(self, ax=None, show_raw=False):
		# Assume linear relationship between speed and duration -> get durations distribution
		# ? + noisy estimate of distance
		# model distance distributions
		durations = {a:[] for a in self.arms_names}
		speeds = {a:[] for a in self.arms_names}
		_, distances = self.get_pathlength_estimates()

		# estimate duration
		for i, arm in self.arms_params.iterrows():
			for s in random.choices(self.trials_speeds, k=1000):
				distance = distances[arm["name"]].rvs(size=1)
				durations[arm["name"]].append((distance/s)/30)
				speeds[arm["name"]].append(s)

		# plot
		if ax is None: 
			f, ax = plt.subplots(figsize=(8, 8))
			legend = True
		else:
			legend=False
		for arm in durations.keys():
			ax.scatter(speeds[arm], durations[arm], label=arm, color=self.colors[arm], alpha=.2)

		if show_raw:
			self.plot_duration_over_speed_raw(ax=ax)

		ax.set(facecolor=[.2, .2, .2], xlim=[0, 8], ylim=[0, 10], xlabel="speed (a.u.)", ylabel="duration (s)")

		if legend: ax.legend()

		return ax

	# TODO probs names should have the same names of the arms distances
	def simulate_trials(self, n_trials=10000, ax=None, distances=None, plot=True):
		# For N trials draw a random sped
		# for each arm estimate the escape duration given noisy estimate of length + noisy estimate of duration

		# get distances distribution
		if distances is None:
			_, distances = self.get_pathlength_estimates()

		if self.experiment_name == "symmetric":
			exp_distances = [distances["rightmedium"], distances["rightmedium"]]
			arms_names = ["left_medium", "right_medium"]
		elif self.experiment_name == "asymmetric":
			exp_distances = [distances["leftfar"], distances["rightmedium"]]
			arms_names = ["left_far", "right_medium"]
		else:
			exp_distances = [distances["centre"], distances["leftfar"], distances["rightmedium"]]
			arms_names = ["centre", "left_far", "right_medium"]

		# simulate
		outcomes = []
		for i in range(n_trials):
			# get random speed and apply noise
			s = random.choice(self.trials.mean_speed.values)
			est_speeds = [s + np.random.normal(0, self.speed_noise_param, 1) for i in exp_distances]

			# get random length
			est_lengths  = [d.rvs(size=1) for d in exp_distances]

			#calc duration
			fastest = np.argmin([l/s for l,s in zip(est_lengths, est_speeds)])
			outcomes.append(arms_names[fastest])

		probs = {arm:calc_prob_item_in_list(outcomes, arm) for arm in arms_names}
		# print(probs)

		if plot:
			if ax is None:
				f,ax = plt.subplots()

			x = np.arange(len(arms_names))
			cc = [self.colors2[arm] for arm in arms_names]
			
			ax.bar(x, probs.values(), color=cc)
			ax.set(xticks=x, xticklabels=arms_names, title=self.experiment_name, facecolor=[.2, .2, .2])
			ax.tick_params(axis='x', rotation=45)

		return probs

	def plot_durationspeed_distribution(self, ax=None, original_colors=True, 
										distances=None, distances_modelled=None):
		if ax is None:
			make_legend = True
			f, ax = plt.subplots(figsize=(8, 8))
		else:
			make_legend = False

		if not original_colors:
			colors = {a:"k" for a in self.distances.keys()}
		else:
			colors = self.colors

		if distances is None and distances_modelled is None:
			distances, distances_by_exp, model_dists = self.get_pathlength_from_rawdata()

		for arm in distances.keys():
			ax.hist(distances[arm], color=colors[arm], label=arm, alpha=.5, density=True, bins=5)
			d =  model_dists[arm]
			x = np.linspace(d.ppf(0.01), d.ppf(.99), 100)
			ax.plot(x, d.pdf(x), color=colors[arm], lw=3)

		if make_legend:
			ax.legend()
			ax.set(title=self.experiment_name,  facecolor=[.2, .2, .2], xlim=[200, 900], ylim=[0, .015], xlabel="duration * speed", ylabel="p")

		return ax

	def psychometric_curve_uncertainty(self):
		self.keep_trials_experiment("asymmetric")

		# get p(R) at different noise factors
		pathlength_noise_factors = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60]
		probs = []
		for factor in pathlength_noise_factors:
			self.pathlength_noise_factor = factor
			self.get_pathlength_estimates(plot=False)
			res = self.simulate_trials(plot=False, n_trials=1000)
			probs.append(res["right_medium"])

		# Fit sigmoid and poly curve + plot
		popt, pcov = curve_fit(half_sigmoid, pathlength_noise_factors, probs)
		fitted = polyfit(3, pathlength_noise_factors, probs)

		x = np.linspace(1, pathlength_noise_factors[-1])
		fitted_sigmoid = half_sigmoid(x, *popt)

		self.plot_psychometric(x, fitted, fitted_sigmoid, pathlength_noise_factors, probs)

	def psychometric_curve_distances(self):
		# Get distances curvesfor extra arms
		self.keep_trials_experiment("asymmetric")
		moments, distances = self.get_pathlength_estimates()
		del moments["centre"]
		del distances["centre"]

		n_intermediate_arms = 8
		# intermediate_lengths = np.linspace(moments["rightmedium"][0], moments["leftfar"][0], n_intermediate_arms+2)[1:-1]
		intermediate_lengths = np.linspace(0, 900, n_intermediate_arms+2)[1:-1]

		intermediate_arms = {"intermediate{}".format(i+1):stats.norm(loc=v, scale=np.square(v)*self.pathlength_noise_factor)
							for i,v in enumerate(intermediate_lengths)}

		distance_distributions = [v for v in {**distances, **intermediate_arms}.values()]
		means = [v.mean() for v in distance_distributions]
		means = np.sort(np.array(means))

		# simulate one arm at the time vs rightmedium
		probs=[]
		for dist in means:
			test_dists = dict(rightmedium=distances["rightmedium"], leftfar=stats.norm(loc=dist, scale=math.sqrt(dist)*self.pathlength_noise_factor))
			res = self.simulate_trials(plot=False, n_trials=10000, distances=test_dists)
			probs.append(res["right_medium"])

		# Fit sigmoid and poly curve + plot
		popt, pcov = curve_fit(sigmoid, means, probs, method='dogbox', p0=[1000, 0.001]) # TODO make this work  , 
		fitted = polyfit(3, means, probs)

		x = np.linspace(means[0], means[-1], 100)
		fitted_sigmoid = sigmoid(x, *popt)

		ax = self.plot_psychometric(x, means, probs, label="p(R) over len(L)", sigmoid=fitted_sigmoid)
		ax.set(xlim=[0, 2*distances["rightmedium"].mean()], xlabel="path length")

		real_probs = self.simulate_trials(plot=False)

		for arm, d in moments.items():
			ax.axvline(d[0], color=self.colors[arm])
		
		return means, probs 
	

	def plot_psychometric(self, x, x_data, y_data, sigmoid=None, label=None, ployfit=None):
		# curve
		f, ax = plt.subplots(figsize=(8, 8))
		ax.scatter(x_data, y_data, color="white", label=label)

		if ployfit is not None: ax.plot(x, ployfit(x), color="k", label="fitted")
		if sigmoid is not None: ax.plot(x, sigmoid, color="k", label="fitted")

		for xx in x_data:
			ax.axvline(xx, color="w", lw=.5, alpha=.1)
		for yy in y_data:
			ax.axhline(yy, color="w", lw=.5, alpha=.1)

		# ax.axhline(.5, color="g", lw=2, alpha=.5)

		ax.set(ylim=[-0.01, 1.01], facecolor=[.2, .2, .2], ylabel="p")
		ax.legend()
		return ax




a = Analyzer("asymmetric")

# %%
means, probs = a.psychometric_curve_distances()

# %%
# run stuff
a.keep_trials_experiment("asymmetric")
# ax = a.plot_duration_over_speed_raw(original_colors=True)
# ax2 = a.plot_duration_over_speed_modelled(show_raw=False)
ax, raw_dists_asym, colors2_asym = a.plot_durationspeed_distribution()

a.keep_trials_experiment("threearms")
# a.plot_duration_over_speed_raw(ax=ax, original_colors=True)
# ax2 = a.plot_duration_over_speed_modelled(ax=ax2, show_raw=False)
ax, raw_dists_3arms, colors2_3arms = a.plot_durationspeed_distribution()


# %%
# Test
a.keep_trials_experiment("asymmetric")
a.get_pathlength_estimates(plot=True)
a.simulate_trials(n_trials=10000)


#%%
# test
def sigmoid(x, x0, k):
	 y = 1 / (1 + np.exp(-k*(x-x0)))
	 return y

# xdata = np.array([0.0,   1.0,  3.0, 4.3, 7.0,   8.0,   8.5, 10.0, 12.0])
# ydata = np.array([0.01, 0.02, 0.04, 0.11, 0.43,  0.7, 0.89, 0.95, 0.99])
xdata = means
ydata = probs


popt, pcov = curve_fit(sigmoid, xdata, ydata, p0=[1000, 0.001], )
print(popt)

x = np.linspace(-1, 15, 50)
y = sigmoid(x, *popt)

plt.plot(xdata, ydata, 'o', label='data', color="r")
plt.plot(x,y, label='fit', color="k")
plt.ylim(0, 1.05)
plt.legend(loc='best')
plt.show()

#%%
