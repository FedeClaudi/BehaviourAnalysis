# %%
import numpy as np
import matplotlib.pyplot as plt
import math
from fcutils.maths.geometry import calc_distance_between_points_2d, calc_angle_between_vectors_of_points_2d
from fcutils.plotting.colors import colorMap
from tqdm import tqdm

# %%
# ------------------------------- Define funcs ------------------------------- #
def get_path_length_from_points(A, B, C):
    BC = calc_distance_between_points_2d(C, B)
    AC = calc_distance_between_points_2d(A, C)
    return BC+AC


def get_path_length(gamma, theta):
    theta = np.radians(theta)

    A = (0, 0)
    B = (0, 1)
    C = (gamma * np.sin(theta), gamma * np.cos(theta))

    return get_path_length_from_points(A, B, C)


def get_deltax_from_points(A, B, C):
    raise NotImplementedError("need to adjust for probability that P is in AC!!")
    path_length = get_path_length_from_points(A, B, C)
    shortc, deltx = [], []
    for i in range(niters):
        P = (np.random.uniform(0, C[0]), np.random.uniform(0, C[1]))
        AP = calc_distance_between_points_2d(A, P)
        PB = calc_distance_between_points_2d(B, P)
        deltx.append(path_length - (AP+PB))
        shortc.append(AP+PB)

    return path_length, shortc, deltx

def get_deltax(gamma, theta):
    raise NotImplementedError("need to adjust for probability that P is in AC!!")
    path_length = get_path_length(gamma, theta)
    theta = np.radians(theta)

    A = (0, 0)
    B = (0, 1)
    C = (gamma * np.sin(theta), gamma * np.cos(theta))
    return get_deltax_from_points(A, B, C)


# %%
# ------------ Visualise path length, shortcut length and delta x ------------ #
f = plt.figure(figsize=(10, 5))
ax = f.add_subplot(121, projection='polar')
ax2 = f.add_subplot(122, projection='polar')


niters = 1000
thetas = np.linspace(0, 360, 101) # angle of initial arm segment
gammas = np.linspace(0.1, 2, 11) # length of initial arm segment

p_shortcut = .1

for gamma in tqdm(gammas):
    path_length = np.zeros_like(thetas)
    delta_x = np.zeros((len(thetas), niters))
    shortcut = np.zeros((len(thetas), niters))

    for n, theta in enumerate(thetas):
        pl, sc, dx = get_deltax(gamma, theta)
        path_length[n] = pl
        delta_x[n, :] = dx
        shortcut[n, :] = sc

    color = colorMap(gamma, name='Greens', vmin=-2, vmax=np.max(gammas)+2)

    ax.plot(np.radians(thetas), path_length, label='Path length', color=color)
    ax2.plot(np.radians(thetas), p_shortcut*np.mean(shortcut, axis=1)+(1-p_shortcut)*path_length, 
                            color=color)

ax.set(yticks=[1, 2, 3], #ylim=[0, 1.5],
            xticks=np.radians([0, 90, 180, 270]), title='path and shortcut lengths')
ax2.set(yticks=[1,2,3], #ylim=[0, 1.5],
            xticks=np.radians([0, 90, 180, 270]), title='Delta X')
f.tight_layout()


# %%
# ------------------ Render delta x as spatial visualisation ----------------- #
niters = 100

A = (0, 0)
B = (0, 1)

x = np.linspace(-1, 1, 101) # points lattice
y = np.linspace(-1, 1, 101)

values = np.zeros((len(x), len(x)))
for i, _x in enumerate(x):
    for ii, _y in enumerate(y):
        C = (_x, _y)
        path_length, shortc, deltx = get_deltax_from_points(A, B, C)
        values[ii, i] = np.mean(deltx)


# Plot
f, ax = plt.subplots(figsize=(8, 8))

ax.imshow(values, extent=[-1, 1, -1, 1], origin='lower', cmap='Greens')

ax.scatter(*A, s=150, color='k', label='A', zorder=99)
ax.scatter(*B, s=150, color='g', label='B', zorder=99)

ax.set(xlim=[-1, 1], ylim=[-1, 1])
ax.legend()

plt.show()
# %%
# ----------------------- Plot deltax as theta vs gamma ---------------------- #
thetas = np.linspace(0, 360, 101) # angle of initial arm segment
gammas = np.linspace(0.1, 2, 101) # length of initial arm segment

values = np.zeros((len(thetas), len(gammas)))
for i, t in enumerate(thetas):
    for ii, g in enumerate(gammas):
        path_length, shortc, deltx = get_deltax(g, t)
        values[i, ii] = np.mean(deltx)

# Plot
f, ax = plt.subplots(figsize=(8, 8))

ax.imshow(values, extent=[0, 2*np.pi, np.min(gammas), np.max(gammas)], origin='lower', cmap='Greens')



ax.set(xlim=[0, 2*np.pi], ylim=[np.min(gammas), np.max(gammas)])
ax.legend()

plt.show()

# %%
