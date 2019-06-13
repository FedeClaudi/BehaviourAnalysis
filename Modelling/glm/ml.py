# %%
from Utilities.imports import *
from Modelling.glm.glm_data_loader import GLMdata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler as Scaler
import statsmodels.formula.api as smf
from patsy import dmatrices
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz


# %% 
# get data
glm = GLMdata(load_trials_from_file=True)

d = namedtuple("d", "all asym sym")
data = d(glm.trials, glm.asym_trials, glm.sym_trials)

# %%
# Extract more info on the data
right_rois, left_rois = [18, 6, 13], [17, 8, 11, 2, 3, 12, 22]

# Remove trials from sessinos with insufficient exploration
cleaned = []
for k,v in data._asdict().items():
    cleaned.append(v.loc[(v.total_travel > 10000) & (v.tot_time_in_shelter < 650)])
data = d(*cleaned)



# %%
# Extract metrics from expl
for k,v in data._asdict().items():
    v["mean_expl_speed"] = [np.nanmean(dd.tracking_data_exploration[:, 2]) for i, dd in v.iterrows()]
    v["delta_theta"] = 135 - v.iTheta.values

    # TODO make it cleaner
    v["time_on_left_exp"] = [np.sum(np.isin(dd.tracking_data_exploration[:, -1], left_rois)) for i, dd in v.iterrows()]
    v["time_on_right_exp"] = [np.sum(np.isin(dd.tracking_data_exploration[:, -1], right_rois)) for i, dd in v.iterrows()]

    # TODO this doesnt work
    v["rel_time_on_right"] = v["time_on_left_exp"].values / v["time_on_right_exp"].values
    v["correct_rel_time_on_right"] = v.rel_time_on_right.values / v.rLen.values

# %% 
# Etxtract metrics from out tracking
for k,v in data._asdict().items():
    v["out_trip_duration"] = [dd.outward_tracking_data.shape[0] for i, dd in v.iterrows()]
    v["out_trip_mean_speed"] = [np.mean(dd.outward_tracking_data[:, 2]) for i,dd in v.iterrows()]
    v["out_trip_iAngVel"]  = [np.sum(np.abs(calc_ang_velocity(calc_angle_between_points_of_vector(dd.outward_tracking_data[:, :2]))))/dd.out_trip_duration
                                 for i,dd in v.iterrows()]

# %%
# drop columns
cols_to_keep = ['session_uid',  'time_out_of_t', 'x_pos', 'experiment_asymmetric',
       'y_pos', 'speed',  'escape_right', 
       'origin_right',
       'total_travel', 'tot_time_in_shelter', 'tot_time_on_threat', 
       'median_vel', 'rLen', "out_trip_iAngVel", 
       'iTheta', 'mean_expl_speed', 'delta_theta', 'time_on_left_exp', "out_trip_mean_speed", 
       'time_on_right_exp', 'rel_time_on_right', 'correct_rel_time_on_right', "out_trip_duration"]

kept = []
for k,v in data._asdict().items():
    kept.append(v[cols_to_keep])
data = d(*kept)

# %%
# do
"""
    73%: eq = "escape_right ~  origin_right + x_pos + y_pos + speed + rel_time_on_right + mean_expl_speed" 


"""


eq = "escape_right ~  origin_right + x_pos + y_pos + speed + rel_time_on_right" 
mean_corrects = []
for i in range(200):
    train, test = train_test_split(data.all, test_size=.25)
    Y, X = dmatrices(eq, data=train, return_type='matrix')
    terms  = X.design_info.column_names

    X      = np.asarray(X) # fixed effect
    Y      = np.asarray(Y).flatten()

    Y_test, X_test = dmatrices(eq, data=test, return_type='matrix')
    Y_test, X_test = np.asarray(Y_test), np.asarray(X_test)

    # ! Decision tree classifier
    tree = DecisionTreeClassifier(max_depth=2)

    tree.fit(X,Y)


    # Test
    y_test_pred = tree.predict(X_test)
    corr = np.sum((y_test_pred == Y_test.ravel()))
    # print("Correctly classified: {}/{} - {}%".format(corr, len(Y_test), round(corr/len(Y_test), 2)*100))
    mean_corrects.append(round(corr/len(Y_test), 2)*100)
print("Overall: {}% +- {}  correct".format(np.mean(mean_corrects), round(np.std(mean_corrects), 2)))

#%%
dot_data = export_graphviz(tree, out_file=None, 
                    feature_names=terms,  
                    class_names=["left", "right"],  
                    filled=True, rounded=True,  
                    )  
graph = graphviz.Source(dot_data)  
graph 


#%%
