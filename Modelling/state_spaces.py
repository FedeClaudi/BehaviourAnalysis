# %%
# Imports
import sys
sys.path.append("./")

from Utilities.imports import *
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold, datasets, decomposition, discriminant_analysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# %matplotlib inline  

# %%
# Define class
class StateSpacesAnalyser:
    def __init__(self):
        # Load the tracking data for each session
        # sessions = Session.fetch("uid")
        # ? uid restricted to fetch only one
        self.data = pd.DataFrame(((Session & "experiment_name='PathInt2'") * (TrackingData.BodyPartData & "bpname='body'")).fetch()) # ! fetching only one for speed
        self.data_tail = pd.DataFrame(((Session & "experiment_name='PathInt2'") * (TrackingData.BodyPartData & "bpname='tail_base'")).fetch()) # ! fetching only one for speed
        self.stimuli = pd.DataFrame(((Session & "experiment_name='PathInt2'") * (Stimuli)).fetch())

        print("Got data")

    def compute_variables(self):
        print("processing")

        shelter = [500, 710]

        # Get speed and angular velocity at each frame for each session
        x, y, speeds, ang_speeds, shelter_dist, xy, orientation, shelter_speed, y_speed = [], [], [], [], [], [], [], [], []
        for (i, brow), (ii, trow) in tqdm(zip(self.data.iterrows(), self.data_tail.iterrows())):
            # Timepoints with speed > 4*std are considered tracking errors
            speed = brow.tracking_data[:, 2].copy()
            speed_th = np.mean(speed) + 4*np.std(speed)

            speeds.append(line_smoother(speed))

            dshelt = calc_distance_from_shelter(brow.tracking_data[:, :2], shelter)
            shelter_speed.append(line_smoother(np.append(np.diff(dshelt), 0)))
            x.append(line_smoother(brow.tracking_data[:, 0]))
            y.append(line_smoother(brow.tracking_data[:, 1]))
            
            # y speed
            y_speed.append(np.append(np.diff(line_smoother(brow.tracking_data[:, 1])), 0))

            # ? Body axis orientation
            angles = line_smoother(calc_angle_between_vectors_of_points_2d(brow.tracking_data[:, :2].T, trow.tracking_data[:, :2].T))
            orientation.append(angles)
            ang_speeds.append(np.append(np.abs(np.diff(angles)), 0))

            # if i == 1: break


        self.session = pd.DataFrame.from_dict(dict(s=np.hstack(speeds), o=np.hstack(orientation), ang_speeds=np.hstack(ang_speeds), y_speed=np.hstack(y_speed)))
        a = 1
        # self.data["speed"] = speeds
        # self.data["shelter_speed"] = shelter_speed
        # self.data["orientation"] = orientation
        # self.data["ang_speed"] = ang_speeds



    def plot_trials(self):
        fig = plt.figure(figsize = (16, 16))
        ax = fig.add_subplot(111, projection='3d')

        speed = self.data.tracking_data.values[0][:, 2]
        x, y, z = self.data.tracking_data.values[0][:, 0], self.data.tracking_data.values[0][:, 1], self.data.orientation.values[0] # self.data.tracking_data.values[0][:, 3]
        x, y, z, speed = line_smoother(x)-500, line_smoother(y)-500, line_smoother(z), line_smoother(speed)

        # ax.plot(x, y, z, color="k", alpha=.1)
        ax.scatter(x, y, z, c=speed, alpha=.3, vmax=10)

        for i, stim in self.stimuli.iterrows():
            if stim.overview_frame == -1: continue

            delay = 400
            ax.plot(x[stim.overview_frame:stim.overview_frame+delay], 
                    y[stim.overview_frame:stim.overview_frame+delay], 
                    z[stim.overview_frame:stim.overview_frame+delay], 
                    alpha=.8)

            ax.scatter(x[stim.overview_frame], y[stim.overview_frame], z[stim.overview_frame], 
                    color="g", s=500)
            ax.scatter(x[stim.overview_frame+delay],  y[stim.overview_frame+delay],  z[stim.overview_frame+delay], 
                    color="b", s=500)

        ax.set(xlabel="x", ylabel="y", zlabel="orientation", facecolor=[.2, .2, .2], )


        plt.show()
        a = 1

    def plot_statespace(self):
        # For now plot for just one session

        fig = plt.figure(figsize = (16, 16))
        ax = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        # x, y, z =self.data.speed.values[0], self.data.shelter_speed.values[0], self.data.tracking_data.values[0][:, 1]
        xp, yp, s = self.data.tracking_data.values[0][:, 0], self.data.tracking_data.values[0][:, 1], self.data.tracking_data.values[0][:, 2]
        xp, yp, s = line_smoother(xp), line_smoother(yp), line_smoother(s)
        s = np.append(np.diff(yp), 0) * s
        ax2.plot(xp, yp, s, color="k", alpha=.1)

        ax2.set(title="Tracking data", xlabel="x pos", ylabel="y pos", zlabel="Speed", facecolor=[.2, .2, .2])

        x, y, z =self.data.speed.values[0], self.data.shelter_speed.values[0], self.data.ang_speed.values[0]
        x, y, z = line_smoother(x), line_smoother(y), line_smoother(z)


        ax.plot(x, y, z, color="k", alpha=.1)

        ax.set(title="state space", xlabel="speed", ylabel="shelter_speed", zlabel="ang_speed", facecolor=[.2, .2, .2],
                ylim=[-6, 6], xlim=[0, 8])

        # plot stimuli
        for i, stim in self.stimuli.iterrows():
            if stim.overview_frame == -1: continue

            print(stim)
            delay = 400
            ax.plot(x[stim.overview_frame:stim.overview_frame+delay], 
                    y[stim.overview_frame:stim.overview_frame+delay], 
                    z[stim.overview_frame:stim.overview_frame+delay], 
                    alpha=.8)

            ax2.plot(xp[stim.overview_frame:stim.overview_frame+delay], 
                yp[stim.overview_frame:stim.overview_frame+delay], 
                s[stim.overview_frame:stim.overview_frame+delay], 
                alpha=.8)

            ax.scatter(x[stim.overview_frame], y[stim.overview_frame], z[stim.overview_frame], 
                    color="g", s=500)
            ax.scatter(x[stim.overview_frame+delay],  y[stim.overview_frame+delay],  z[stim.overview_frame+delay], 
                    color="b", s=500)



        plt.show()
        a = 1

    def pca_dem_data(self):
        print("Doing stuffsies")
        # standardize variables and fit PCA
        x = StandardScaler().fit_transform(self.session.values)

        # pca
        pca = PCA(n_components=4)
        principalComponents = pca.fit_transform(x)
        df = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2', "pc3", "pc4"])



        # t sne
        # tsne = manifold.TSNE(n_components=2, init='pca').fit_transform(x[:10000, :])
        # f,ax = plt.subplots()
        # ax.scatter(tsne[:, 0], tsne[:, 1], c=stim_times[:10000], cmap="Reds")
        # ax.set(facecolor=[.2, .2, .2])

        # Get the timepoints corresponding to escapes
        stim_starts = [s for s in self.stimuli.overview_frame.values if s > 0  ]
        stim_times = np.zeros(len(self.session))

        for start in stim_starts:
            stim_times[start: start+150] = 1
                    # stim_times[start:start + 40] = 1
                    # stim_times[start + 40:start + 80] = 2
                    # stim_times[start + 80:start + 120] = 3
                    # stim_times[start + 120:start + 160] = 4
                    # stim_times[start + 160:start + 200] = 5
        df["escape"] = stim_times

        # Plot
        # f, ax = plt.subplots()
        fig = plt.figure(figsize = (16, 16))
        ax = fig.add_subplot(111, projection='3d')
        time = np.arange(len(df))

        # ax.scatter(df.pc1.loc[df.escape==0], df.pc2.loc[df.escape==0], df.pc3.loc[df.escape==0],  c="w", alpha=.1)
        for i, start in enumerate(stim_starts):
            if start < 0 or start > len(df): continue
            # ax.plot(df.pc1.values[start:start+150], df.pc2.values[start:start+150], df.pc3.values[start:start+150], lw=5)
            ax.scatter(df.pc1.values[start:start+150], df.pc2.values[start:start+150], df.pc3.values[start:start+150], c=np.arange(150), s = 100, alpha=.4, cmap="Reds")
            # if i == 3: break


        ax.set(title="PCA", xlabel="pc1", ylabel="pc2", facecolor=[.2, .2, .2], xlim=[-2, 12], ylim=[-3, 7])

        plt.show()


        a = 1


# %%
# run
# an = StateSpacesAnalyser()
# an.compute_variables()
# an.plot_statespace()


# %%
if __name__ == "__main__":
    an = StateSpacesAnalyser()
    an.compute_variables()
    an.pca_dem_data()

    plt.show()
