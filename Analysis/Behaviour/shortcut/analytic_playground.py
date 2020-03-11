# %%
import numpy as np
import matplotlib.pyplot as plt
import math
from fcutils.maths.geometry import calc_distance_between_points_2d


# %%
# Define points

A = (0, 0)
B = (0, 1)

x = np.linspace(-1, 1, 101) # points lattice
y = np.linspace(-1, 1, 101)

# %%
def get_expected_delta_x(A, B, C):
    AB = calc_distance_between_points_2d(A, B)
    AC = calc_distance_between_points_2d(A, C)
    BC = calc_distance_between_points_2d(B, C)

    return (AC + BC - AB)/AC




# %%
values = np.zeros((len(x), len(x)))

for i, _x in enumerate(x):
    for ii, _y in enumerate(y):
        C = (_x, _y)
        values[ii, i] = get_expected_delta_x(A, B, C)


# %%
# Plot
f, ax = plt.subplots(figsize=(8, 8))

ax.imshow(values, extent=[-1, 1, -1, 1], origin='lower', cmap='Greens', vmin=0, vmax=2)

ax.scatter(*A, s=150, color='k', label='A', zorder=99)
ax.scatter(*B, s=150, color='g', label='B', zorder=99)

ax.set(xlim=[-1, 1], ylim=[-1, 1])
ax.legend()

plt.show()
# %%
# Partial derivative of expected delta x wrt theta.
def eval_partial(gamma, theta):
    theta = np.radians(theta)
    numer = np.sin(theta)
    denum = ((gamma + math.sqrt(gamma**2 +1 -2*np.cos(theta)))**2)*(math.sqrt(gamma**2 +1 -2*np.cos(theta)))
    return numer/denum


gamma = 2
theta = np.linspace(0, 360, 1001)
deltax = [eval_partial(gamma, t) for t in theta]
r = [1 for I in deltax]

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
ax.plot(np.radians(theta), deltax)

# %%
