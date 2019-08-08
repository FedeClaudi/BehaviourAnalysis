# %%
# imports
import sys
sys.path.append("./")
from Utilities.imports import *
import scipy.io
from scipy.spatial.distance import euclidean as distance

%matplotlib inline

"""
    Test script to load and inspect data from 8 neuropixel mice from the cortex lab
    https://figshare.com/articles/Eight-probe_Neuropixels_recordings_during_spontaneous_behaviors/7739750
"""

# TODO load data from other two mice
# %%
# Vars
one_mouse = False

# ? Load spikes data
spikes_folder = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Ephys_Datasets/spks"
mice =['Krebs','Waksman','Robbins']
tstarts = {'Krebs':3811,'Waksman':3633,'Robbins':3323}

frate = 30 # represent time in frames


# %%
# Data loading functions
def load_parsed_data():
    print("loading data")
    data = {}
    for f in os.listdir(spikes_folder):
        if ".npy" in f:
            data[f.split(".")[0]] = np.load(os.path.join(spikes_folder, f))
    return data


def parse_cortexlab_data():
    for mouse, tstart in tstarts.items():
        spikes = scipy.io.loadmat(os.path.join(spikes_folder,"spks{}_Feb18.mat".format(mouse)))["spks"]

        # spikes["st"] -> spike times in seconds  Has 8 arrays, one for each probe
        # spikes["clu"] -> clusted identity of each spike in st
        # spikes["Wheights"]) -> height of each cluster on the probe

        # Extract spike times and create a matrix neurons x time
        data = {}
        for probe in np.arange(8):
            print("Processing {} - probe {}".format(mouse, probe))
            clu = spikes["clu"][0][probe]
            sf = np.floor(spikes["st"][0][probe] * frate).astype(np.int32) # convert time in 
            # height = spikes["Wheights"][probe]

            clusters = np.unique(clu)
            maxt = np.max(sf) - tstart * frate

            # Create a matrix of timexcluster to contain all spikes from all units
            matrix = np.zeros((max(clusters)+1, maxt+1)).astype(np.int8)

            for t, c in zip(sf, clu):
                matrix[c, t - tstart * frate] = 1

            np.save(os.path.join(spikes_folder, "{}_probe_{}.npy".format(mouse, probe)), matrix)
        
        if one_mouse: break

# %%
# Load data
data = load_parsed_data()

# %%
# Calculate stuff

dists, ddts, nn = [], [], []
for i, (p, mtx) in enumerate(data.items()):
    dotdelta = np.zeros(mtx.shape[1]) #  real numbers with dot product between subsequent vecotrs
    dist = np.zeros(mtx.shape[1]) 

    for t in np.arange(mtx.shape[1]):
        if t == 0: 
            pass
        else:
            ddt = np.dot(mtx[:, t], mtx[:, t-1])
            d = distance(mtx[:, t], mtx[:, t-1])
            dotdelta[t], dist[t]  = ddt, d

    dists.append(dist), ddts.append(dotdelta), nn.append(mtx.shape[0])

# %%
# corr data
corr_data = [[], [], []]
for dist, dotdelta, n in zip(dists, ddts, nn):
    corr_data[0].append(n)
    corr_data[1].append(np.mean(dist))
    corr_data[2].append(np.mean(dotdelta))

# %%
# Plot stuff
f, axarr = create_figure(subplots=True, ncols=8, nrows=3, sharex=True, sharey=True, figsize=(20, 16))

for i, (dist, dotdelta, n )in enumerate(zip(dists, ddts, nn)):
    axarr[i].plot(dist, color=grey, lw=.5, alpha=.7)
    axarr[i].plot(line_smoother(dist, window_size=1001), color=red, lw=2)
    axarr[i].set(title="{}n".format(n), ylim=[0, 20]) # , ylabel="time dot product", xlabel="time (frames)")

#%%
f, ax = plt.subplots(figsize = (12, 12))
sns.regplot(corr_data[0], corr_data[1], robust=True, color=green, ax=ax)
ax.set(xlabel="# neurons", ylabel="mean $dist(A(t), A(t-1))$")
#%%
