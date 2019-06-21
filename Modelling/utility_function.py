# %%
from Utilities.imports import *
%matplotlib inline  
from Modelling.glm.glm_data_loader import GLMdata



# . %% Get data
# glm = GLMdata(load_trials_from_file=True)
# params = glm.load_maze_params()
# params.index = params.name

# # TODO get arms measurements in CM
# # need to convert the lengths from the two expeirments to be on the same scale
# conv_fact = 28.368  # ratio of asym.right.len and sym right len in the params df
# params["adjusted_length"] = np.divide(params.length.values, conv_fact) # ! this wont be correct for all experiments

# params


#%%
# Evaluate single arms using geodesic gradient
from Modelling.maze_solvers.gradient_agent import GradientAgent as GeoAgent
folder = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/analysis_metadata/maze_solvers/single_arms"

agent = GeoAgent(grid_size=500, start_location=[255, 100], goal_location=[255, 345])

arms_data = dict(name=[], n_steps=[], distance=[], iTheta=[])
for arm in os.listdir(folder):
    if not "jpg" in arm: continue
    # ? get maze, geodesic and walk
    agent.maze, agent.free_states = agent.get_maze_from_image(model_path=os.path.join(folder, arm))
    agent.geodesic_distance = agent.geodist(agent.maze, agent.goal_location)

    walk = agent.walk()
    # f, ax = plt.subplots()
    # agent.plot_walk(walk, ax=ax)
    # f.savefig(os.path.join(folder, arm.split(".")[0]+"_walk.png"))

    # ? evalueate walk
    arms_data["name"].append(arm.split(".")[0].split("_")[1])
    arms_data["n_steps"].append(len(walk))
    arms_data["distance"].append(round(np.sum(calc_distance_between_points_in_a_vector_2d(np.array(walk)))))
    # arms_data["iTheta"].append(np.sum(np.abs(calc_ang_velocity(calc_angle_between_points_of_vector(np.array(walk))[1:]))))
    arms_data["iTheta"].append(int(arm.split("_")[0]))    


params = pd.DataFrame(arms_data)
params["cost"] = params.distance.values * params.iTheta.values
params = params.sort_values("cost")
params




# %%
# Define utility functions and plotter function

def constant_product(l=None, t=None, x=None, y=None, k=None):
    # indifference curve: x*y = k
    # -> y = k/x
    if l is not None and t is not None:
        return l*t
    elif x is not None and k is not None:
        return k/x
    elif x is not None and y is not None:
        return x*y.T

def x_squared_it(l=None, t=None, x=None, y=None, k=None):
    # indifference curve: x^2*y = k
    # -> y = k/x^2
    if l is not None and t is not None:
        return (l**2)*t
    elif x is not None and k is not None:
        return k/x**2
    elif x is not None and y is not None:
        return np.multiply(x,x)*y.T

def x_power_it(l=None, t=None, x=None, y=None, k=None, power=3):
    # indifference curve: x^2*y = k
    # -> y = k/x^2
    if l is not None and t is not None:
        return (l**power)*t
    elif x is not None and k is not None:
        return k/np.power(x, power)
    elif x is not None and y is not None:
        return np.power(x, power)*y.T

def plotter_dataframe(func, x_max=50, y_max=50, xvar="distance", yvar="n_steps"):
    # Get equilibria values
    arms_points = {arm:[l, t] for arm, l, t in zip(params.name, params[xvar], params[yvar])}
    arms_equilibria = {arm:func(l=l, t=t) for arm, l, t in zip(params.name, params[xvar], params[yvar])} # l*t
    equils = sorted(list(arms_equilibria.values()))
    colors = get_n_colors(20)

    # Get background image
    x = np.tile(np.linspace(0, x_max, 500), 500).reshape(500, 500)
    img = func(x=x, y=x)  # x.T*y  # value at all locations
    x = np.linspace(0, x_max, 100)

    f, ax = plt.subplots(figsize=(16, 12))
    for c, eq in zip(colors, arms_equilibria.values()):
        y = func(x=x, k=eq)
        ax.plot(x, y, color=c, alpha=.8, lw=2)

    # plot vertical and horizontal lines
    for x in params[xvar]:
        ax.axvline(x, linestyle="--", color="w", lw=.15)
    for t in params[yvar]:
        ax.axhline(t, linestyle="--", color="w", lw=.15)

    # Plot equilibria points
    for c, (k, (xx, yy)) in zip(colors, arms_points.items()):
            ax.scatter(xx, yy, s=300, label=k, color=c, alpha=.6)

    # show background
    ax.imshow(img, interpolation="nearest", origin="lower", extent=(0,  x_max, 0, y_max))

    # set axes
    ax.set(xlabel=xvar, ylabel=yvar)
    ax.legend()

    return f, ax, arms_equilibria


def plotter(func, shaded=False, x_max=50):
    # Get vars
    arms = ["asym_right", "asym_left",  "mbv2_long", "mbv2_short"]
    lengths, ithetas = [19, 26.085,  45, 28], [180, 270, 360, 180]
    # lengths, ithetas = [0.72, 1,   1, 0.622], [180, 270, 360, 180] # ? rLen
    # lengths, ithetas = [19, 26.085,  45, 28], [45, 45,  90, 45] # ? atheta
    # lengths, ithetas = [19, 26.085,  45, 28], [180, 225,  360, 225] # ? atheta + iTheta
    # lengths, ithetas = [19, 26.085,  45, 28], [90, 135,  90, 90] # ? max angle on path
    arms_points = {arm:[l, t] for arm, l, t in zip(arms, lengths, ithetas)}
    colors = ["red", "red", "orange", "orange"]

    # Get equilibria values
    arms_equilibria = {arm:func(l=l, t=t) for arm, l, t in zip(arms, lengths, ithetas)} # l*t
    equils = sorted(list(arms_equilibria.values()))

    # Get background image
    x = np.tile(np.linspace(0, x_max, 500), 500).reshape(500, 500)
    img = func(x=x, y=x)  # x.T*y  # value at all locations

    x = np.linspace(0, x_max, 100)

    f, ax = plt.subplots()
    #  plot indifference curve
    if shaded:
        for k in np.arange(equils[0], equils[1], 10):
            y = func(x=x, k=k)
            ax.plot(x, y, color="red", alpha=.025)

        for k in np.arange(equils[2], equils[-1], 10):
            y = func(x=x, k=k)
            ax.plot(x, y, color="orange", alpha=.025)

    for c, eq in zip(colors, arms_equilibria.values()):
        y = func(x=x, k=eq)
        ax.plot(x, y, color=c)

    # plot vertical and horizontal lines
    for x in lengths:
        ax.axvline(x, linestyle="--", color="w", lw=.15)
    for t in ithetas:
        ax.axhline(t, linestyle="--", color="w", lw=.15)

    # Plot equilibria points
    for c, (k, (xx, yy)) in zip(colors, arms_points.items()):
            ax.plot(xx, yy, 'o', label=k, color=c)

    # # Get values at different points on idiff curve given the equilibria
    # for angle in [45, 90, 135, 180, 225, 270]:
    #     l = arms_equilibria["asym_right"]/angle
    #     print("For  an arm of length = right medium: angle {} - length {}".format(angle, l))
    #     ax.scatter(l, angle, color="r", s=100, alpha=.5)

    # show background
    ax.imshow(img, interpolation="nearest", origin="lower", extent=(0,  x_max, 0, 400))

    # set axes
    ax.set(xlabel="arm length (a.u.)", ylabel="iTheta")
    ax.legend()

    return f, ax, arms_equilibria
#%%
f, ax, arms_equilibria = plotter_dataframe(constant_product, x_max=500, y_max=900, 
                                            xvar="distance", yvar="distance")
f.savefig(os.path.join(folder, "utilityspace.png"))
# arms_equilibria= plotter(constant_product, x_max=50)









#%%
# explore actual trials data
glm = GLMdata(load_trials_from_file=True)
trials = glm.asym_trials
trials["escape_duration_frames"] = [t.shape[0] for t in trials.tracking_data_trial.values]
trials["log_escape_duration_s"] = np.log(np.array(trials.escape_duration_frames.values / trials.fps.values, np.float32))
trials["escape_distance"] = [np.sum(calc_distance_between_points_in_a_vector_2d(t[:, :2])) for t in trials.tracking_data_trial.values]
left = trials.loc[trials.escape_arm == "Left_Far"]
right = trials.loc[trials.escape_arm == "Right_Medium"]
all_expl_speed = np.hstack([t[:, 2] for t in trials.tracking_data_exploration.values])

# %%
f, axarr = plt.subplots(nrows=2, sharex=False, figsize=(8, 6))
axarr[0].scatter(left.mean_speed, left.log_escape_duration_s ,  c="r", alpha=.75)
axarr[0].scatter(right.mean_speed, right.log_escape_duration_s,  c="b", alpha=.75)
# sns.regplot(left.mean_speed, left.escape_duration_s,  color="green", ax=axarr[0],
#             order=1, truncate=True, ci=95)
# sns.regplot(right.mean_speed, right.escape_duration_s,  color="white", ax=axarr[0],
#             order=1, truncate=True, )
axarr[0].set(facecolor="k", xlabel="speed a.u.", ylabel="log(duration s)", xlim=[3, 10])


axarr[1].hist(all_expl_speed, density=True, bins=20)
axarr[1].set(facecolor="k", xlabel="speed . a . u", ylabel="density")

#%%

median_dist = [np.median(left.escape_distance.values), np.median(right.escape_distance.values)]
conv_factor = params.loc[params.name == "rightmedium"].distance.values / median_dist[1]
median_dist = [x*conv_factor for x in median_dist]

#%%
l = left.median()
r = right.median()

f,ax = plt.subplots()
x = np.linspace(0, 800, 100)

ax.plot(x, constant_product(x=x, k=l.escape_distance*conv_factor*l.log_escape_duration_s), "r")
ax.plot(x, constant_product(x=x, k=r.escape_distance*conv_factor*r.log_escape_duration_s), "b")

ax.scatter(l.escape_distance * conv_factor, l.log_escape_duration_s, color="r")
ax.scatter(r.escape_distance * conv_factor, r.log_escape_duration_s, color="b")

ax.set(xlim=[0, 800], ylim=[0,5])


#%%
