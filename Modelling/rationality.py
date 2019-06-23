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
_, left_intercept, left_coff, left_reg = linear_regression(left.mean_speed.values, left.log_escape_duration_s.values)
# y = p2*x + c -> duration = -0.85*speed + 9.3  
_, right_intercept, right_coff, right_reg = linear_regression(right.mean_speed.values, right.log_escape_duration_s.values)
print("Left: ", round(left_coff, 2), round(left_intercept, 2), "Right: ", round(right_coff, 2), round(right_intercept, 2))


# %%
# For N trials draw a random speed and get the duration of escape on both arms
speeds, left_durs, right_durs = [], [], []
for i in range(100):
    speed = np.random.normal(loc=mean_speed, scale=std_speed, size=1)
    speeds.append(speed)
    left_durs.append(left_coff*speed + left_intercept)
    right_durs.append(right_coff*speed + right_intercept)


f, ax = plt.subplots()
ax.scatter(left.mean_speed.values, left.log_escape_duration_s.values, marker="v", color='r')
ax.scatter(right.mean_speed, right.log_escape_duration_s, marker="v", color='b')
ax.scatter(speeds, left_durs, color='r', alpha=.5)
ax.scatter(speeds, right_durs, color='b', alpha=.5)

ax.plot(x, speed_dist.pdf(x), color='k')



#%%
# Now for N trials take two speeds, measure the duration of each side and see which one is faster
trials = []
for i in range(10000):
    lspeed, rspeed = np.random.normal(loc=mean_speed, scale=std_speed, size=1), np.random.normal(loc=mean_speed, scale=std_speed, size=1)
    ldur, rdur = left_coff*lspeed + left_intercept, right_coff*rspeed + right_intercept
    if rdur <= ldur: trials.append(1)
    else: trials.append(0)

print("p(R): ", round(np.mean(trials), 2))


# %%
# Now for N trials draw M speeds per side and simulate the expected cost (duration) then compare
N, M = 1000, 2
trials = []
for i in range(N):
    lspeeds, rspeeds = np.random.normal(loc=mean_speed, scale=std_speed, size=M), np.random.normal(loc=mean_speed, scale=std_speed, size=M)
    ldur = np.mean([left_coff*speed + left_intercept for speed in lspeeds])
    rdur = np.mean([right_coff*speed + right_intercept for speed in rspeeds])
    if rdur < ldur: trials.append(1)
    else: trials.append(0)

print("p(R): ", round(np.mean(trials), 2))
#%%
