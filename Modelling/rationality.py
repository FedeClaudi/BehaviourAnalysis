"""
Given the mice's distribution of escape speeds and escape path lengths on the 2 arms experiments>
For a range of speeds calculate the value of each path -> expected duration
Then multiply these values by their probability

Then simulate N trials in which you draw two random speed, one per arm, and compare the values you'd expect



"""

# %%
import sys
sys.path.append("./")
from Utilities.imports import *
from Modelling.glm.glm_data_loader import GLMdata

from scipy import stats

data = GLMdata

%matplotlib inline

#%%
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

# %%
# ! remove trials whose escape distance is too high
trials = trials.loc[trials.escape_distance < 1000]
left = trials.loc[trials.escape_arm == "Left_Far"]
right = trials.loc[trials.escape_arm == "Right_Medium"]

# Get escape speed distribution
# (left, right)
median_distance = (np.median(left.escape_distance), np.median(right.escape_distance))
mean_speed = np.mean(trials.mean_speed)
std_speed = np.std(trials.mean_speed)
speed_dist = stats.norm(loc=mean_speed, scale=std_speed)
x = np.linspace(3, 10, num=100)


# %%
f, ax = plt.subplots()
# TODO compare distribution of speed with raw data
ax.plot(x, speed_dist.pdf(x))
ax.hist(trials.mean_speed, density=True)

#%%
# Fit a linera regression to speed, duration so that we can calculate the expected duration given a speed
_, left_intercept, left_coff, left_reg = linear_regression(left.mean_speed.values, eft.escape_duration_s.values)
# y = p2*x + c -> duration = -0.85*speed + 9.3  
_, right_intercept, right_coff, right_reg = linear_regression(right.mean_speed.values, right.escape_duration_s.values)
print("Left: ", round(left_coff, 2), round(left_intercept, 2), "Right: ", round(right_coff, 2), round(right_intercept, 2))


# %%
# For N trials draw a random speed and get the duration of escape on both arms
speeds, left_durs, right_durs = [], [], []
for i in range(100):
    speed = np.random.normal(loc=mean_speed, scale=std_speed, size=1)
    speeds.append(speed)
    left_durs.append(left_coff*speed + left_intercept)
    right_durs.append(right_coff*speed + right_intercept)


f, axarr = plt.subplots(ncols=2)
axarr[0].scatter(left.mean_speed.values, left.escape_duration_s.values, marker="v", color='r')
axarr[0].scatter(right.mean_speed, right.escape_duration_s, marker="v", color='b')
axarr[1].hist(right.escape_duration_s, bins=10, orientation="horizontal", color="b", alpha=.3, density=True)
axarr[1].hist(left.escape_duration_s, bins=10, orientation="horizontal", color="r", alpha=.3, density=True)

# ax.scatter(speeds, left_durs, color='r', alpha=.5)
# ax.scatter(speeds, right_durs, color='b', alpha=.5)




#%%
# Now for N trials take two speeds, measure the duration of each side and see which one is faster
trials_ = []
for i in range(10000):
    lspeed, rspeed = np.random.normal(loc=mean_speed, scale=std_speed, size=1), np.random.normal(loc=mean_speed, scale=std_speed, size=1)
    ldur, rdur = left_coff*lspeed + left_intercept, right_coff*rspeed + right_intercept
    if rdur <= ldur: trials_.append(1)
    else: trials_.append(0)

print("p(R): ", round(np.mean(trials_), 2))


# %%
# Now for N trials draw M speeds per side and simulate the expected cost (duration) then compare
N, M = 1000, 2
trials_ = []
for i in range(N):
    lspeeds, rspeeds = np.random.normal(loc=mean_speed, scale=std_speed, size=M), np.random.normal(loc=mean_speed, scale=std_speed, size=M)
    ldur = np.mean([left_coff*speed + left_intercept for speed in lspeeds])
    rdur = np.mean([right_coff*speed + right_intercept for speed in rspeeds])
    if rdur < ldur: trials_.append(1)
    else: trials_.append(0)

print("p(R): ", round(np.mean(trials_), 2))
# %%
# ? WIP bayesian estiamte of time per arm
# ? Get distribution of times T and model as a Gamma distribution -> p(duration)
shift = np.min(trials.escape_duration_s)
t = trials.escape_duration_s.values -  shift # divide by 10 to keep it in 0-1 range
t_dist = stats.gamma(np.std(t))

# ? plot T distribution
x = np.linspace(0, 10, 100)
f, ax = plt.subplots()
ax.hist(t, bins=30, density=True, color="k", alpha=.3)
ax.plot(x, t_dist2.pdf(x), color="k", lw=3, label="$p(T)$")
ax.legend()

# ? get p(arm|duration)
t_r = right.escape_duration_s - shift
t_l = left.escape_duration_s - shift

t_l_dist = stats.norm(loc=np.mean(t_l), scale=np.std(t_l))
t_r_dist = stats.gamma(2)

p_at_r = [t_r_dist.pdf(z)/(t_l_dist.pdf(z) + t_r_dist.pdf(z)) for z in x]
p_at_l = [t_l_dist.pdf(z)/(t_l_dist.pdf(z) + t_r_dist.pdf(z)) for z in x]

# Plot distribution of escape times per arm
f, ax = plt.subplots()
# ax.hist(t_r, bins=10, color="b", alpha=.3, density=True)
# ax.hist(t_l, bins=10,  color="r", alpha=.3, density=True)

ax.plot(x, t_l_dist.pdf(x), color="r", lw=3)
ax.plot(x, t_r_dist.pdf(x), color="b", lw=3)
ax.plot(x, p_at_r, color="g", lw=3, label="$p(r|t)$")
ax.plot(x, p_at_l, color="orange", lw=3, label="$p(l|t)$")

ax.legend()


# ? now time to get p(t|a) = p(a|t)*p(t)

p_ta_r = [p_at_r[i]*t_dist.pdf(z) for i,z in enumerate(x)]
p_ta_l = [p_at_l[i]*t_dist.pdf(z) for i,z in enumerate(x)]

# fit curves to the posteriors

f,ax = plt.subplots()
ax.plot(x, p_ta_r, color="g", lw=1)
ax.plot(x, p_ta_l, color="orange", lw=1)
ax.hist(t_r, bins=10, color="b", alpha=.3, density=True)
ax.hist(t_l, bins=10,  color="r", alpha=.3, density=True)

# ax.plot(x, t_l_dist.pdf(x), color="r", lw=1, alpha=.5)
# ax.plot(x, t_r_dist.pdf(x), color="b", lw=1, alpha=.5)

# %%
# ? simulate escapes given rationality and the posterior from bayes

_trials = []
for i in range(10000):
    # draw randomly from the two posteriors
    l, r = random.choice(p_ta_l), random.choice(p_ta_r)
    if r<=l: _trials.append(1)
    else: _trials.append(0)
print("p(R): ", round(np.mean(_trials), 2))


#%%

#%%
# Get distribution of all speeds: escape + exploration
expl_speeds = np.hstack([t[:, 2] for t in trials.tracking_data_exploration])
trials_speeds = np.hstack([t[:, 2] for t in trials.tracking_data_trial])
speeds = np.hstack([expl_speeds, trials_speeds])

f, ax = plt.subplots()
ax.hist(speeds, bins=50, density=True, alpha=.5, color="k")
ax.hist(trials_speeds, bins=50, density=True, alpha=.5, color="m")
ax.hist(trials.mean_speed, density=True, alpha=.4)

ax.set(xlim=[0, 10])





#%%
"""
    Given the linear relationship between speed and and duration, estimate the distribution
    of duration for each arm, given the distribution of speeds
"""
r_durations, l_durations = [], []
for i in range(10000):
    speed = random.choice(trials_speeds)
    ldur, rdur = left_coff*speed + left_intercept, right_coff*speed + right_intercept
    r_durations.append(rdur)
    l_durations.append(ldur)

er_durations, el_durations = [], []
for i in range(10000):
    speed = random.choice(expl_speeds)
    ldur, rdur = left_coff*speed + left_intercept, right_coff*speed + right_intercept
    er_durations.append(rdur)
    el_durations.append(ldur)

f, ax = plt.subplots()
ax.hist(r_durations, color="b", alpha=.5)
# ax.hist(er_durations, color="purple", alpha=.5)
ax.hist(l_durations, color="r", alpha=.5)
# ax.hist(el_durations, color="orange", alpha=.5)

#%%
# ? model the distribution of durations per arm
f,ax = plt.subplots()
x = np.linspace(.5, 8, 100)
right_dist = stats.norm(loc=np.mean(right.escape_duration), scale=np.std(right.escape_duration))
left_dist = stats.norm(loc=np.mean(left.escape_duration), scale=np.std(left.escape_duration))

delta = [r-l for r,l in zip(right_dist.rvs(size=10000), left_dist.rvs(size=10000))]

# ax.hist(trials.escape_duration, bins=10,  color="k", alpha=.3, density=True)
ax.hist(left.escape_duration, bins=10, color="r", alpha=.3, density=True)
ax.hist(right.escape_duration, bins=10,  color="b", alpha=.3, density=True)
ax.plot(x, right_dist.pdf(x), color="b", lw=3)
ax.plot(x, left_dist.pdf(x), color="r", lw=3)

f, ax = plt.subplots()
ax.hist(delta)

ax.set(xlim=[-2, 10])

#%%
test =[1 if random.choice(delta) < 0 else 0 for i in range(10000)]
print(np.mean(test))

#%%
