# %%
import sys
sys.path.append("./")
from Utilities.imports import *
from Modelling.glm.glm_data_loader import GLMdata
from Modelling.maze_solvers.gradient_agent import GradientAgent as GeoAgent

from scipy import stats

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


		
		mean_arm_distances = {"leftfar": np.mean(self.left.escape_distance),
								"rightmedium": np.mean(self.right.escape_distance),
								"centre": np.mean(self.centre.escape_distance)}
		conv_factors = {arm: d / self.arms_params.loc[self.arms_params["name"] == arm].distance.values[0] for arm,d in mean_arm_distances.items()}


		mean_right_dist = np.mean(self.right.escape_distance)
		conv_fact = mean_right_dist / self.arms_params.loc[self.arms_params["name"] == "rightmedium"].distance.values[0]
		self.arms_params["distance"] = [d*conv_factors[n] for d,n in zip(self.arms_params["distance"], self.arms_params["name"])]

	def get_pathlength_estimates(self, plot=False):
		distances = {arm["name"]:stats.norm(loc=arm.distance, scale=math.sqrt(arm.distance)*self.pathlength_noise_factor) for i,arm in self.arms_params.iterrows()}

		if plot:
			f, ax = plt.subplots()
			for a, d in distances.items():
				x = np.linspace(d.ppf(0.01), d.ppf(0.99), 100)
				ax.plot(x, d.pdf(x), color=self.colors[a], label=a, lw=3)
			ax.set(facecolor=[.2, .2, .2], xlabel="length (a.u.)", ylabel="p")
			ax.legend()
		
		return distances

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
		distances = self.get_pathlength_estimates()

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

	def simulate_trials(self, n_trials=10000, ax=None):
		# For N trials draw a random sped
		# for each arm estimate the escape duration given noisy estimate of length + noisy estimate of duration

		# get distances distribution
		distances = self.get_pathlength_estimates()
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

		if ax is None:
			f,ax = plt.subplots()

		x = np.arange(len(arms_names))
		cc = [self.colors2[arm] for arm in arms_names]
		
		ax.bar(x, probs.values(), color=cc)
		ax.set(xticks=x, xticklabels=arms_names, title=self.experiment_name, facecolor=[.2, .2, .2])
		ax.tick_params(axis='x', rotation=45)



a = Analyzer("asymmetric")

# %%
# run stuff
a.keep_trials_experiment("asymmetric")
ax = a.plot_duration_over_speed_raw(original_colors=True)
ax2 = a.plot_duration_over_speed_modelled(show_raw=False)

a.keep_trials_experiment("threearms")
a.plot_duration_over_speed_raw(ax=ax, original_colors=True)
ax2 = a.plot_duration_over_speed_modelled(ax=ax2, show_raw=False)



#%%
# Plot p(R) for raw data
from Processing.trials_analysis.all_trials_loader import Trials

trial_data = Trials(exp_mode=0)
trial_data.plot_parm_experiment()




#%%
# plot p(R) for simulations
f,axarr = plt.subplots(figsize=(10, 8), ncols=3, sharey=True)

a.pathlength_noise_factor = 10
a.speed_noise_param =  1.75

a.get_pathlength_estimates(plot=True)

for ax, exp in zip(axarr, a.experiments_names):
	a.keep_trials_experiment(exp)
	a.simulate_trials(ax=ax, n_trials=10000)

#%%
