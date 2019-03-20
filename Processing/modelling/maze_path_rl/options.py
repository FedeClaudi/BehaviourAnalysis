import sys
sys.path.append('./')

from copy import deepcopy
import numpy as np
import PyQt5
import os
import pickle

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from Processing.tracking_stats.math_utils import *

FLAG_load_model = True


if not FLAG_load_model:
    maze_designs = ['rightmedium.png', 'leftfar.png'] # , 'mb_central.png', 'mb_right.png']
    grid_size = 120

    walks = dict(
    name= [],
    walk= [],
    walk_length=[],
    distance = [],
    tot_distance = [],
    angle = [],
    angle_diff = [],
    tot_angle_diff = [],
    )
    for maze_design in maze_designs:
        print("\n\n\n", maze_design)

        model_fld = "Processing/modelling/maze_path_rl/models"
        name = maze_design.split('.')[0]
        modelname = os.path.join(model_fld, name + '.pkl')

        with open(modelname, 'rb') as read:
            model = pickle.load(read)

        # Get the shortest path using geodesic distances
        walk = np.vstack(model.geodesic_walk())

        walk_length = len(walk)
        walk_distance = calc_distance_between_points_in_a_vector_2d(walk)
        walk_tot_distance = np.sum(walk_distance)

        walk_angle = calc_angle_between_points_of_vector(walk)
        walk_angle_diff = np.insert(0, 0, np.diff(walk_angle))
        walk_tot_angle_diff = np.nansum(np.abs(walk_angle_diff))

        walks['name'].append(name)
        walks['walk'].append(walk)
        walks['walk_length'].append(walk_length)
        walks['distance'].append(walk_distance)
        walks['tot_distance'].append(walk_tot_distance)
        walks['angle'].append(walk_angle)
        walks['angle_diff'].append(walk_angle_diff)
        walks['tot_angle_diff'].append(walk_tot_angle_diff)
    options = pd.DataFrame.from_dict(walks)
    with open('Processing\modelling\maze_path_rl\options.pkl', 'wb') as output:
        pickle.dump(options, output, pickle.HIGHEST_PROTOCOL)
else:
    with open('Processing\modelling\maze_path_rl\options.pkl', 'rb') as read:
        options = pickle.load(read)






with open('Processing\modelling\maze_path_rl\models\PathInt2.pkl', 'rb') as read:
    model = pickle.load(read)



print(options)
colors = get_n_colors(4)
colors = [colors[0], colors[-1]]

f, axarr = plt.subplots(ncols=3)
axarr[0].imshow(model.env.maze_image, cmap='gray', interpolation=None)
for_trials = []
for i, (name, walk, d) in enumerate(zip(options['name'].values, options['walk'].values, options['tot_distance'].values)):
    axarr[0].scatter(walk[:, 0], walk[:, 1], alpha=.75, c=colors[i])

    var = 2*math.sqrt(d)
    action_samples = np.random.normal(loc=d, scale=var, size=10000)
    axarr[1].hist(action_samples, bins=100, alpha=.75, color=colors[i])

    for_trials.append((name, d, var))

l, r = 0, 0
for i in np.arange(1000):
    vals = [np.random.normal(loc=d, scale=var, size=1) for name, d, var in for_trials]

    if np.argmin(vals) == 0:
        r += 1
    else:
        l += 1
    a = 1

x = np.arange(len(for_trials))
y = [r/(l+r), l/(l+r)]
axarr[2].bar(x, y, color=colors)
axarr[2].set(xticklabels=['right', 'left'])
plt.show()





