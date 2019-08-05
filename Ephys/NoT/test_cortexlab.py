import sys
sys.path.append("./")
from Utilities.imports import *
import scipy.io

"""
    Test script to load and inspect data from 8 neuropixel mice from the cortex lab
    https://figshare.com/articles/Eight-probe_Neuropixels_recordings_during_spontaneous_behaviors/7739750
"""

# TODO load data from other two mice

class Analyzer:
    # ? Load spikes data
    spikes_folder = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Ephys_Datasets/spks"

    def __init__(self):
        self.data = {}

    def load_parsed_data(self):
        print("loading data")
        for f in os.listdir(self.spikes_folder):
            if ".npy" in f:
                self.data[f.split("_")[1].split(".")[0]] = np.load(os.path.join(self.spikes_folder, f))

    def parse_cortexlab_data(self):
        data = scipy.io.loadmat(os.path.join(self.spikes_folder,"spksKrebs_Feb18.mat"))
        spikes = data["spks"]

        # spikes["st"] -> spike times in seconds  Has 8 arrays, one for each probe
        # spikes["clu"] -> clusted identity of each spike in st
        # spikes["Wheights"]) -> height of each cluster on the probe

        # Extract spike times and create a matrix neurons x time
        stall = np.zeros((5000, 5500)).astype(np.int8)
        ij, maxt, Wh, iprobe, brainLoc = 0, 0, [], [], []
        frate = 30 

        f, axarr = create_figure(subplots=True, ncols=2, nrows=2)

        data = {}
        for probe, ax in enumerate(axarr):
            if probe == 8: continue
            clu = spikes["clu"][0][probe]
            sf = np.floor(spikes["st"][0][probe] * frate).astype(np.int32) # convert time in 
            # height = spikes["Wheights"][probe]

            clusters = np.unique(clu)
            maxt = np.max(sf)

            # Create a matrix of timexcluster to contain all spikes from all units
            matrix = np.zeros((max(clusters)+1, maxt+1)).astype(np.int8)

            for t, c in zip(sf, clu):
                matrix[c, t] = 1

            ax.imshow(matrix[:, :200])
            ax.set(title="Probe {}".format(probe), xlabel="Time (30fps)", ylabel="Cluster #")
            np.save(os.path.join(self.spikes_folder, "probe_{}.pkl".format(probe)), matrix)
        plt.show()

    def show_matrixes(self):
        f, axarr = create_figure(subplots=True, ncols=4, nrows=2)

        for i, (p, mtx) in enumerate(self.data.items()):
            axarr[i].plot(mtx.T)

            axarr[i].set(title="Probe {}".format(i), xlabel="Time (30fps)", ylabel="Cluster #")


    def inspect(self):
        for i, (p, mtx) in enumerate(self.data.items()):
            delta = np.zeros(mtx.shape)  # vectors that represent the difference between psi t and psy t-1
            dotdelta = np.zeros(mtx.shape[1]) #  real numbers with dot product between subsequent vecotrs

            for t in np.arange(mtx.shape[1]):
                if t == 0: delta[:, t] = 0
                else:
                    dt = mtx[:, t] - mtx[:, t-1]
                    ddt = np.dot(mtx[:, t], mtx[:, t-1])
                    delta[:, t], dotdelta[t] =  dt, ddt
                    a = 1


            f, ax = plt.subplots()
            ax.plot(dotdelta[120000: ])

if __name__ == "__main__":
    a = Analyzer()
    # a.parse_cortexlab_data()
    a.load_parsed_data()
    # a.show_matrixes()
    a.inspect()
    plt.show()
