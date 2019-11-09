# %%
import sys
sys.path.append('./')

from Utilities.imports import *

import networkx as nx

try: from scipy.special import softmax
except: pass
from mpl_toolkits.mplot3d import Axes3D

from Modelling.maze_solvers.agent import Agent

from scipy.special import softmax

%matplotlib inline

# %%
# Get agent
agent = Agent(grid_size=1000, maze_design="PathInt2_old.png")
agent.maze = agent.maze[::-1, :]


# %%
# Define nodes coords etc
coordinates = dict(
    catwalk = (500, 175), 
    threat=(500, 325),
    l_m1 = (120, 700), l_m4=(300, 520), 
    r = (700, 520), r_m1_flip = (900, 325),
    shelter = (500, 700)
)

M1 = [["catwalk", "threat"], ["threat", "l_m1"],
                ["threat", "r"],
                ["l_m1", "shelter"], ["r", "shelter"]]

M4 = [["catwalk", "threat"], ["threat", "l_m4"], ["l_m4", "shelter"],
                ["threat", "r"], ["r", "shelter"]]

M6 = [["catwalk", "threat"], 
        ["threat", "l_m4"], ["l_m4", "l_m1"], ["l_m1", "shelter"],
        ["threat", "r_m1_flip"], ["r_m1_flip", "r"], ["r", "shelter"],]

maze = M1
nodes_names = set([item for sublist in maze for item in sublist])
nodes = {k:coordinates[k] for k in nodes_names}

edges_lengths = [calc_distance_between_points_2d(coordinates[p1], coordinates[p2]) for p1, p2 in maze]
edges = [(n1, n2, w) for (n1, n2), w in zip(maze, edges_lengths)]


# Create graphg 
G=nx.Graph()
G.add_nodes_from(nodes_names)
G.add_weighted_edges_from(edges)


# for each node get the shortest path to shelter length and euclidean distance from shelter
max_geo_dist = nx.shortest_path_length(G, "catwalk", "shelter", weight="weight")
max_euc_dist = calc_distance_between_points_2d(coordinates['catwalk'], coordinates['shelter'])
for node in G.nodes:
    # geodesic distance
    d =  nx.shortest_path_length(G, node, "shelter", weight="weight")
    G.node[node]['geodesic_distance'] = d/max_geo_dist

    # euclidean distance
    ed = calc_distance_between_points_2d(coordinates[node], coordinates['shelter'])
    G.node[node]['euclidean_distance'] = ed/max_euc_dist

for node, data in G.nodes(data=True):
    print("{} -- Geo. dist: {} - eucl. distance: {}\n".format(node, round(data['geodesic_distance'],2), 
                                                round(data['euclidean_distance'], 2)))


# plot graph 
nx.draw(G, with_labels = True, pos=nodes, with_weights=False)

# plot with weights (geo distance)
# nx.draw(G, with_labels = True, with_weights=True)


# %%
# Simulate a trial
def step(G, pos, prev_pose):
    options = [p for p in list(G.neighbors(pos)) if p != prev_pose]
    # costs = np.array([nx.shortest_path_length(G, option, goal, weight="weight") for option in options])
    weights = [1 - G.node[n]['euclidean_distance'] for n in options]
    return pos, random.choices(options, weights=weights, k=1)[0]

n_routes, routes = 5000, []
for i in range(n_routes):
    pos = "catwalk"
    prev_pose = ""
    goal = "shelter"
    nsteps = 0
    route = []
    while pos != goal:
        # print("step {} at {}".format(nsteps, pos))
        prev_pose, pos = step(G, pos, prev_pose)
        nsteps +=1 
        route.append(pos)
    # print("step {} at {}".format(nsteps, pos))
    routes.append(route)


r =len([r for r in routes if "r" in r[-2]])/n_routes
print("\np(R): {}".format(r))


# %%


# %%
