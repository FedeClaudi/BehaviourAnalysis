# %%
# Imports
import sys
sys.path.append("./")

from Utilities.imports import *
from Processing.analyse_experiments import ExperimentsAnalyser

from matplotlib.patches import Circle, Wedge, Polygon

# %matplotlib inline

ea = ExperimentsAnalyser()
data = ea.load_trials_from_pickle()
# del data["maze2"]
# del data["maze3"]


# %%
fig, axarr = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True)
axarr = axarr.flatten()
# plt.axis("equal")
xx, rr = [68, 36, 68, 36], [12, 14, 12, 14]
for data_n, (k, d) in enumerate(data.items()):
    #%%
    # get XY tracking data
    # trdata = np.vstack([t.tracking_data[:int(t.time_out_of_t * t.fps)]
    #                     for i,t in d.iterrows()])[:, :2].astype(np.int32)
    trdata = np.vstack([t.tracking_data for i,t in d.iterrows()])[:, :2].astype(np.int32)

    trdata[:, 0] -= np.min(trdata[:, 0])
    trdata[:, 1] -= np.min(trdata[:, 1])
    trdata = np.int32(trdata / 5)
    x_centre, y_centre = int(max(trdata[:, 0])/2), int(max(trdata[:, 1])/2)

    # get XY tracking data shifted by 1 in time
    shifted_trdata = np.zeros_like(trdata)
    shifted_trdata[0:-1, :] =  trdata[1:, :]

    # %%
    # Get the angle between the two sets of data for eac frame
    angles = calc_angle_between_vectors_of_points_2d(trdata.T, shifted_trdata.T)

    #%%
    # Get average angle at each location
    count, tot_angle = np.zeros((np.max(trdata[:, 0])+1, np.max(trdata[:, 1])+1)), np.zeros((np.max(trdata[:, 0])+1, np.max(trdata[:, 1])+1))
    sins, coss = np.zeros((np.max(trdata[:, 0])+1, np.max(trdata[:, 1])+1)), np.zeros((np.max(trdata[:, 0])+1, np.max(trdata[:, 1])+1))
    for (x, y), a in zip(trdata, angles):
        count[x, y] += 1
        tot_angle[x, y] += a
        sins[x, y] += np.sin(math.radians(a))
        coss[x, y] += np.cos(math.radians(a))

    # theta = np.deg2rad(tot_angle / count)
    theta = np.zeros_like(sins)
    for i, x in enumerate(np.arange(sins.shape[0])):
        for ii, y in enumerate(np.arange(sins.shape[1])):
            if i % 2 == 0 and ii % 2 == 0:
                theta[x, y] = math.atan(sins[x, y]/coss[x, y])
        
    theta[theta == 0] = np.nan

    # %%
    # Get as 5d array with x, y, theta, sin(theta), cos(theta)
    xytsc = np.vstack([[x, y, np.radians(theta[x, y]), np.sin(theta[x, y]), math.cos(theta[x, y])] for x in np.arange(theta.shape[0]) for y in np.arange(theta.shape[1])])

    # %%
    # Quiver plot +  scatter
    q = axarr[data_n].quiver(xytsc[:, 0], xytsc[:, 1], xytsc[:, 3], xytsc[:, 4], xytsc[:, 2], angles="xy", scale_units='width', 
                        scale=50, cmap="Spectral", alpha=1, headaxislength=5)
    axarr[data_n].set(title=k)

    circle = Circle((xx[data_n], 35), rr[data_n], color="r", fill=False, lw=4)
    axarr[data_n].add_artist(circle)

    f2, ax = plt.subplots(figsize=(10, 10))
    q = ax.quiver(xytsc[:, 0], xytsc[:, 1], xytsc[:, 3], xytsc[:, 4], xytsc[:, 2], angles="xy", scale_units='width', 
                        scale=50, cmap="Spectral", alpha=1, headaxislength=5)
    ax.set(title=k, xlim=[0, 150], ylim=[0, 120])

    
plt.show()

#%%
