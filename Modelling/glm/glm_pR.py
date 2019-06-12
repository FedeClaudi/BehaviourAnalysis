# %%
from Utilities.imports import *
from Modelling.glm.glm_data_loader import GLMdata

# %% 
# get data
glm = GLMdata(load_trials_from_file=True)
data = glm.trials
features = sorted(data.columns)
data_descr = data.dtypes

demo = glm.trials[["experiment_asymmetric", "escape_right", "iTheta", "rLen",]]


#%%
# fit the glm
eq = "escape_right ~  iTheta*rLen + median_vel "
model, res, y, predictions = glm.run_glm(GLM.trials, eq)


# plot the results
color_label = data.experiment_asymmetric.values+1
glm.plotter(y, predictions, color_label, logistic=False)

#%%
"""
['duration',
 'escape_arm',
 'escape_duration',
 'escape_left',
 'escape_right',
 'experiment_asymmetric',
 'experiment_name',
 'experiment_symmetric',
 'exploration_id',
 'exploration_start',
 'fps',
 'iTheta',
 'max_speed',
 'mean_speed',
 'median_vel',
 'origin_arm',
 'origin_arm',
 'origin_left',
 'origin_right',
 'rLen',
 'session_number_trials',
 'session_uid',
 'speed',
 'time_out_of_t',
 'tot_time_in_shelter',
 'tot_time_on_threat',
 'total_angular_displacement',
 'total_travel',
 'tracking_data_exploration',
 'tracking_data_trial',
 'trial_id',
 'x_pos',
 'y_pos']



"""

#%%


#%%


#%%


#%%
