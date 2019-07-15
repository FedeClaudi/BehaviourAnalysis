# %%
# Imports
import sys
sys.path.append("./")

from Utilities.imports import *
from Processing.analyse_experiments import ExperimentsAnalyser

%matplotlib inline

ea = ExperimentsAnalyser()
data = ea.load_trials_from_pickle()["maze1"]

#%%
# get XY tracking data
# trdata = np.vstack([t.tracking_data[:int(t.time_out_of_t * t.fps)]
#                     for i,t in data.iterrows()])[:, :2].astype(np.int32)
trdata = np.vstack([t.tracking_data for i,t in data.iterrows()])[:, :2].astype(np.int32)

trdata[:, 0] -= np.min(trdata[:, 0])
trdata[:, 1] -= np.min(trdata[:, 1])
trdata = np.int32(trdata / 5)

# get XY tracking data shifted by 1 in time
shifted_trdata = np.zeros_like(trdata)
shifted_trdata[0:-1, :] =  trdata[1:, :]

# %%
# Get the angle between the two sets of data for eac frame
angles = calc_angle_between_vectors_of_points_2d(trdata.T, shifted_trdata.T)
print(min(angles), max(angles), np.mean(angles))
#%%
# Plot scatterplot
f, ax = plt.subplots()
# ax.scatter(trdata[:, 0], trdata[:, 1], c=angles, vmin=0, vmax=360, s=2, alpha=.5)


#%%
# Get average angle at each location
count, tot_angle = np.zeros((np.max(trdata[:, 0])+1, np.max(trdata[:, 1])+1)), np.zeros((np.max(trdata[:, 0])+1, np.max(trdata[:, 1])+1))

for (x, y), a in zip(trdata, angles):
    count[x, y] += 1
    tot_angle[x, y] += a
theta = np.deg2rad(tot_angle / count)
# plt.imshow(np.rot90(theta))

# %%
# Get as 3d array
# xyt = np.vstack([[x, y, theta[x, y]] for x,y in trdata])

xyt = np.vstack([[x, y, math.radians(theta[x, y])] for x in np.arange(theta.shape[0]) for y in np.arange(theta.shape[1])])

f, ax = plt.subplots()
plt.scatter(xyt[:, 0], xyt[:, 1], c=xyt[:, 2], s=2)
#%%
# Quiver plot
u, v = np.cos(xyt[:, 2], where=~np.isnan(xyt[:, 2])), np.sin(xyt[:, 2], where=~np.isnan(xyt[:, 2]))


fig, ax = plt.subplots()
plt.axis("equal")
q = ax.quiver(xyt[:, 0], xyt[:, 1], u, v, xyt[:, 2], angles='xy', scale_units='xy', 
                    scale=.11, cmap="Reds", alpha=.5)

#%%
