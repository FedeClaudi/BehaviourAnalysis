# %%
import networkx as nx
import matplotlib.pyplot as plt
import numpy.random as npr
import numpy as np
import math
from tqdm import tqdm
import seaborn as sns

from fcutils.maths.geometry import calc_distance_between_points_2d


# %%

f, ax = plt.subplots()

ax.axvline(400)
ax.axvline(600)

ax.axhline(700)
ax.axhline(300)
ax.set(xlim=[0, 1000], ylim=[0, 1000])

def get_random_coords():
    coords = dict(
        shelter = (500, 700),
        threat=(500, 300),
        L = (npr.randint(0, 400), npr.randint(300, 700)),
        R = (npr.randint(600, 1000), npr.randint(300, 700)),
    )
    return coords

arms = [
    ('shelter', 'L'), ('L', 'threat'),
    ('shelter', 'R'), ('R', 'threat'),
]



def get_random_graph():
    nodes = get_random_coords()
    edges_lengths = [calc_distance_between_points_2d(nodes[p1], nodes[p2]) for p1, p2 in arms]
    edges = [(n1, n2, w) for (n1, n2), w in zip(arms, edges_lengths)]

    G=nx.Graph()
    G.add_nodes_from(nodes.keys())
    G.add_weighted_edges_from(edges)

    return G, edges, nodes

def sas(x0, x1, cos_theta):
    return math.sqrt(x1**2 + x0**2 + 2*x0*x1*cos_theta)

def get_side_max_eucl(x_p, y_p, y_s, len_a, len_b, len_c, ):
    # Get angle of arm
    cos_theta = (len_c**2 + len_b**2 - len_a**2)/(2*len_b*len_c)

    # GEt max eucl dist.
    eucls = [sas(l, y_s, cos_theta) for l in np.linspace(0, len_b+1, 300)]
    return np.max(eucls)

G, edges, nodes = get_random_graph()
nx.draw(G, with_labels = True, pos=nodes, with_weights=False)

# %%
# for each node get the shortest path to shelter length and euclidean distance from shelter
ratios, eucl_ratios = [], []
for i in tqdm(range(4000)):
    G, edges, nodes = get_random_graph()
    
    l_geodist = np.sum([edge[2] for edge in edges if 'L' in edge])
    r_geodist = np.sum([edge[2] for edge in edges if 'R' in edge])
    ratios.append(l_geodist/r_geodist)

    l_edge_close = [edge for edge in edges if 'L' in edge and 'threat' in edge][0][2]
    l_edge_far = [edge for edge in edges if 'L' in edge and 'shelter' in edge][0][2]
    r_edge_close = [edge for edge in edges if 'R' in edge and 'threat' in edge][0][2]
    r_edge_far = [edge for edge in edges if 'R' in edge and 'shelter' in edge][0][2]
    c = nodes['shelter'][1] - nodes['threat'][1]

    l_eucldist = get_side_max_eucl(*nodes['L'], nodes['shelter'][1],  l_edge_far, l_edge_close, c)
    r_eucldist = get_side_max_eucl(*nodes['R'], nodes['shelter'][1],  r_edge_far, r_edge_close, c)

    eucl_ratios.append(l_eucldist/r_eucldist)

_ = plt.hist(ratios, bins=20, alpha=.5, label='geod', density=True)
_ = plt.hist(eucl_ratios, bins=20, alpha=.5, label='eucl', density=True)
_ = plt.legend()
# %%
f, ax = plt.subplots()
sns.regplot(ratios, eucl_ratios, scatter_kws={'alpha':.5}, line_kws={'lw':3, 'color':'r', 'zorder':99}, ax=ax)
ax.plot([0, 3], [0, 3])

# %%
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(np.array(ratios).reshape(-1, 1), np.array(eucl_ratios).reshape(-1, 1))

# %%
reg._coeff

# %%
