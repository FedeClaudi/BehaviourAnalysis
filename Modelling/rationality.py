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
# mean_speed = (np.mean(left.mean_speed), np.mean(right.mean_speed))
# std_speed = (np.std(left.mean_speed), np.std(right.mean_speed))
mean_speed = np.mean(trials.mean_speed)
std_speed = np.std(trials.mean_speed)
speed_dist = stats.norm(loc=mean_speed, scale=std_speed)

#%%
# Fit a linera regression to speed, duration so that we can calculate the expected duration given a speed
_, left_intercept, left_coff, left_reg = linear_regression(left.mean_speed.values, left.escape_duration_s.values)
# y = p2*x + c -> duration = -0.85*speed + 9.3  
_, right_intercept, right_coff, right_reg = linear_regression(right.mean_speed.values, right.escape_duration_s.values)
print("Left: ", round(left_coff, 2), round(left_intercept, 2), "Right: ", round(right_coff, 2), round(right_intercept, 2))
# %%
# For N trials draw a random speed and get the duration of escape on both arms
f, ax = plt.subplots()
for i in range(100):
    speed = np.random.normal(loc=mean_speed, scale=std_speed, size=1)
    left_dur = 



