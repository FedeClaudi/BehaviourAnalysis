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
fig, axarr = plt.subplots(figsize=(16, 16), ncols=2, nrows=2, sharex=True, sharey=True)
axarr = axarr.flatten()
# plt.axis("equal")

#? Conv fact = 1
xx, rr = [348, 309, 242, 187], [68, 60, 60, 70]
yy = 177

# ? Conv fact = 2
# xx, rr = [172, 121, 123, 93], 28
# yy = 90

# ? Conv fact = 8
# xx, rr = [42, 29, 31, 23], 8
# yy = 24
for data_n, (k, d) in enumerate(data.items()):
    #%%
    # get XY tracking data
    # trdata = np.vstack([t.tracking_data[:int(t.time_out_of_t * t.fps)]
    #                     for i,t in d.iterrows()])[:, :2].astype(np.int32)
    trdata = np.vstack([t.tracking_data for i,t in d.iterrows()])[:, :2].astype(np.int32)

    trdata[:, 0] -= np.min(trdata[:, 0])
    trdata[:, 1] -= np.min(trdata[:, 1])
    trdata = np.int32(trdata / 1)
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
                        scale=65, cmap="Spectral", alpha=.7, headaxislength=5)
    axarr[data_n].set(title=k)

    # Get angles on threat area and take average
    # x, r = xx[data_n], rr[data_n]
    # ttheta = np.vstack([[s,c] for xx,yy,s,c in zip(xytsc[:, 0], xytsc[:, 1], xytsc[:, 3], xytsc[:, 4]) if x-r < xx < x+r and yy-r < yy < yy+r])
    # musin, mucos = np.nanmean(ttheta[:, 0]), np.nanmean(ttheta[:, 1])

    # for ax in axarr:
    #     q = ax.quiver(5, 5, musin, mucos, angles="xy", scale_units='width', 
    #                     scale=4,  color="w", alpha=.4, headaxislength=5)
    # q = axarr[data_n].quiver(5, 5, musin, mucos, angles="xy", scale_units='width', 
    #             scale=4,  color="r", alpha=1, headaxislength=5)
    # circle = Circle((xx[data_n], yy), r, color="w", fill=False, alpha=.2, lw=2)
    # axarr[data_n].add_artist(circle)


    # # f2, ax = plt.subplots(figsize=(10, 10))
    # q = ax.quiver(xytsc[:, 0], xytsc[:, 1], xytsc[:, 3], xytsc[:, 4], xytsc[:, 2], angles="xy", scale_units='width', 
    #                     scale=50, cmap="Spectral", alpha=1, headaxislength=5)
    # ax.set(title=k, xlim=[0, 150], ylim=[0, 120])

    
plt.show()

#%%
