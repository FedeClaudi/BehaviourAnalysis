# %%
import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project
from Utilities.imports import *
%matplotlib inline

from Processing.psychometric_analysis import PsychometricAnalyser
pa = PsychometricAnalyser()

print(pa.paths_lengths)

#%%
# Plot utility function 
fig, ax = create_figure(subplots=False)

# show Psi curve
npoints = 1000
# psi = np.zeros((npoints, npoints))
# for x in np.arange(npoints):
#     for y in np.arange(npoints):
#         psi[y, x] = angle_between_points_2d_clockwise([0, 0], [x, y])

pos = ax.imshow(psi, origin="lower", extent=[0, 1500, 0, 1500], cmap="coolwarm")
# fig.colorbar(pos, ax=ax)


# SHow mazes and IC
ax.scatter([pa.paths_lengths["distance"].values[-1] for i in range(4)], pa.paths_lengths.distance,
                    color=black)
for i, _psi in enumerate(pa.paths_lengths["georatio"].values):
    if i < 3: 
        # ? Fit non linear indifference curve
        ax.plot([0, 1200], [0, _psi*1200], color=black, alpha=0.75)
    else:
        ax.plot([0, 1200], [0, _psi*1200], color=black, alpha=0.75)

ax.axvline(pa.paths_lengths["distance"].values[-1], color=[.2, .2, .2], ls="--")
ax.set(title="$Utility space$", xlim=[0, 1000], ylim=[0, 1000], xlabel="$\mu_R$", ylabel="$\mu_L$")

# Take 3 arbitrary vertical and horizzonalt lines
p = [200, 500, 850]

ortholines(ax, [1, 0], [200, 200], color=green)
ortholines(ax, [1, 0], [500, 500], color=purple)
ortholines(ax, [1, 0], [850, 850], color=orange)

h = [psi[pp, :].T for pp in p]
f2, ax2 = create_figure(subplots=False)

ax.plot(h)

#%%
