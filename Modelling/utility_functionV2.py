# %%
from Utilities.imports import *
%matplotlib inline  
from Modelling.glm.glm_data_loader import GLMdata
from Modelling.maze_solvers.gradient_agent import GradientAgent as GeoAgent
from Modelling.utility_function import plotter_dataframe, constant_product



# %% Get data
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



#%%
f, ax, arms_equilibria = plotter_dataframe(constant_product, params, x_max=5, y_max=1200, 
                                            xvar="torosity", yvar="iTheta")
f.savefig(os.path.join(folder, "utilityspace.png"))


#%%
