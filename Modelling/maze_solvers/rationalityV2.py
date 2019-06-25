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
# explore actual trials data
glm = GLMdata(load_trials_from_file=True)
trials = glm.asym_trials
trials["escape_duration_frames"] = [t.shape[0] for t in trials.tracking_data_trial.values]
trials["escape_duration_s"] = np.array(trials.escape_duration_frames.values / trials.fps.values, np.float32)
trials["log_escape_duration_s"] = np.log(np.array(trials.escape_duration_frames.values / trials.fps.values, np.float32))
trials["escape_distance"] = [np.sum(calc_distance_between_points_in_a_vector_2d(t[:, :2])) for t in trials.tracking_data_trial.values]
left = trials.loc[trials.escape_arm == "Left_Far"]
right = trials.loc[trials.escape_arm == "Right_Medium"]
all_expl_speed = np.hstack([t[:, 2] for t in trials.tracking_data_exploration.values])

# ! remove trials whose escape distance is too high
trials = trials.loc[trials.escape_distance < 1000]
left = trials.loc[trials.escape_arm == "Left_Far"]
right = trials.loc[trials.escape_arm == "Right_Medium"]

#%%
# Get distribution of all speeds: escape + exploration
expl_speeds = np.hstack([t[:, 2] for t in trials.tracking_data_exploration])
trials_speeds = np.hstack([t[:, 2] for t in trials.tracking_data_trial])
speeds = np.hstack([expl_speeds, trials_speeds])

mean_speed_dist = stats.norm(loc = 5.5, scale=1.75)
x = np.linspace(mean_speed_dist.ppf(.01), mean_speed_dist.ppf(.99), 100)

f, ax = plt.subplots()
# ax.hist(speeds, bins=50, density=True, alpha=1, color="k", label="all speed")
# ax.hist(trials_speeds, bins=50, density=True, alpha=.5, color="m", label="trials speed")
ax.hist(trials.mean_speed, density=True, color="w", alpha=.3, label="mean trials speed")
ax.plot(x, mean_speed_dist.pdf(x), color="orange", label="escape speed distribution")

ax.set(xlim=[0, 8], facecolor=[.2, .2, .2], xlabel="speed (a.u.)", ylabel="p")
ax.legend()

# %%
# Get arms params with geo agent
folder = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/analysis_metadata/maze_solvers/good_single_arms"
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


params = pd.DataFrame(arms_data)
params

# %%
# adjust arms legnths to math data 
mean_right_dist = np.mean(right.escape_distance)
conv_fact = mean_right_dist / params.loc[params["name"] == "rightmedium"].distance.values[0]
params["distance"] = params["distance"] * conv_fact


#%%
# Assume linear relationship between speed and duration -> get durations distribution
durations = dict(centre=[], leftfar=[], rightmedium=[])
speeds = dict(centre=[], leftfar=[], rightmedium=[])
for i, arm in params.iterrows():
    for s in random.choices(trials_speeds, k=500):
        durations[arm["name"]].append((arm.distance/s)/30)
        speeds[arm["name"]].append(s)

f, ax = plt.subplots()
for arm, color in zip(durations.keys(), ["m", "r", "g"]):
    ax.plot(sorted(speeds[arm]),sorted(durations[arm])[::-1], label=arm, color=color, alpha=.5)

ax.scatter(left.mean_speed.values, left.escape_duration_s.values, marker="v", color='r')
ax.scatter(right.mean_speed, right.escape_duration_s, marker="v", color='g')

ax.set(facecolor=[.2, .2, .2], xlim=[0, 10], ylim=[0, 15])
ax.legend()
#%%
# Assume linear relationship between speed and duration -> get durations distribution
# ? + noisy estimate of distance
# model distance distributions
colors = {"centre":"m", "leftfar":"r", "rightmedium":"g"}
durations = dict(centre=[], leftfar=[], rightmedium=[])
speeds = dict(centre=[], leftfar=[], rightmedium=[])
distances = {arm["name"]:stats.norm(loc=arm.distance, scale=math.sqrt(arm.distance)*2) for i,arm in params.iterrows()}

f, ax = plt.subplots()
for a, d in distances.items():
    x = np.linspace(d.ppf(0.01), d.ppf(0.99), 100)
    ax.plot(x, d.pdf(x), color=colors[a], label=a, lw=3)
ax.set(facecolor=[.2, .2, .2], xlabel="length (a.u.)", ylabel="p")
ax.legend()

# %%
# plot

for i, arm in params.iterrows():
    for s in random.choices(trials_speeds, k=1000):
        distance = distances[arm["name"]].rvs(size=1)
        durations[arm["name"]].append((distance/s)/30)
        speeds[arm["name"]].append(s)

f, ax = plt.subplots()
for arm in durations.keys():
    # ax.plot(sorted(speeds[arm]),sorted(durations[arm])[::-1], label=arm, color=color, alpha=.5)
    ax.scatter(speeds[arm], durations[arm], label=arm, color=colors[arm], alpha=.5)

ax.scatter(left.mean_speed.values, left.escape_duration_s.values, marker="v", color='k')
ax.scatter(right.mean_speed, right.escape_duration_s, marker="o", color='k')

ax.set(facecolor=[.2, .2, .2], xlim=[0, 8], ylim=[0, 10], xlabel="speed (a.u.)", ylabel="duration (s)")

ax.legend()

#%%
# estimate p(R)

outcomes = []
for i in range(10000):
    s = random.choice(trials.mean_speed.values)
    lspeed, rspeed = s + np.random.normal(0, 1, 1), s + np.random.normal(0, 1, 1)
    l_len, r_len  = distances["leftfar"].rvs(size=1), distances["rightmedium"].rvs(size=1)
    if r_len/rspeed <= l_len/lspeed: outcomes.append(1)
    else: outcomes.append(0)

print("P(R): ", round(np.mean(outcomes), 2))



#%%
f, ax = plt.subplots()
ax.scatter(left.mean_speed.values, left.escape_duration_s.values, marker="v", color='r', label="Left")
ax.scatter(right.mean_speed, right.escape_duration_s, marker="o", color='g', label="Right")
ax.set(facecolor=[.2, .2, .2], xlim=[0, 8], ylim=[0, 10], xlabel="speed (a.u.)", ylabel="duration (s)")
ax.legend()

#%%
