# %%
import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project
from Utilities.imports import *

# from Processing.psychometric_analysis import PsychometricAnalyser
# pa = PsychometricAnalyser()

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

%matplotlib inline

# %%
#  ? Psi and IC

x, y = np.arange(0, 1, .01), np.arange(0, 1, .01)

z = np.zeros((len(x), len(x)))

for i, xx in enumerate(x):
        for ii, yy in enumerate(y):
                if xx == yy:
                        p = .5
                elif xx > yy:
                        p = 1
                else:
                        p = 0
                p = yy - xx  # ! hdxhtcdhgctrcthcgf
                z[i, ii] = p

f, ax = create_figure(subplots=False)

img = ax.imshow(z, alpha=1, aspect="equal", origin="lower", extent=[0, 1, 0, 1], cmap="coolwarm")
f.colorbar(img)


# for k in [.1, .5, 1, 2.5, 10]:
#         ax.plot(x, x*k, color=[.2, .2, .2])

ax.set(title="$\Psi$", xlabel="$d_R$", ylabel="$d_L$", xlim=[0, 1], ylim=[0, 1], xticks=[0, 1], yticks=[0, 1])

#%%
# ? Playing with slope values
f, ax = create_figure(subplots=False)

x = np.arange(0, 1, .01)
y = np.arange(0, 1, .01)

l = len(x)-1
Z = np.zeros((len(x), len(x)))
PSI = np.zeros_like(Z)

for i, xx in enumerate(x):
    for ii, yy in enumerate(y):
        # Z[l-i, ii] = fPsi(xx, yy, k=10)
        # psi = yy/xx
        psi = yy - xx # ! srxdctfyvgbhunjmkl
        PSI[ii, i] = psi
        Z[ii, i] = sigmoid(yy-xx, L=1, x0=0, k=10, b=0)


surf = ax.imshow(Z, extent=[0, 1, 0, 1], cmap=cm.coolwarm, origin="lower", aspect="equal", vmin=0, vmax=1)
f.colorbar(surf)

_x = [.2, .5, .8]

ax.set(title="$\Psi(\\rho)$", xlabel="$d_R$", ylabel="$d_R$")
ax2.set(title="$\psi$", xlabel="$d_R$", ylabel="Y")


#%%
# ? Vertical and horizontal slices are sigmoids
f, axarr = create_figure(subplots=True, ncols=2)
slices = np.arange(5, len(x), 20)

surf = axarr[0].imshow(Z, extent=[0, 1, 0, 1], alpha=.8, cmap=cm.coolwarm, aspect="equal",origin="lower", vmin=0, vmax=1)


for s in slices: 
        axarr[0].axvline(s/100, color=teal)
        axarr[0].axhline(s/100, color=orange)
        axarr[1].plot(Z[s, :], color=orange)
        axarr[1].plot(Z[:, s], color=teal)


axarr[0].set(title="$\Psi(\\rho)$", xlabel="$d_R$", ylabel="$d_R$")



#%%
# ? Show mazes IC
f, ax = create_figure(subplots=False)

x = np.arange(0, 1, .01)
y = np.arange(0, 1, .01)

l = len(x)-1
Z = np.zeros((len(x), len(x)))

for i, xx in enumerate(x):
    for ii, yy in enumerate(y):
        psi = yy/xx
        Z[ii, i] = sigmoid(psi, L=1, x0=1, k=10, b=0)


surf = ax.imshow(Z, extent=[0, 1, 0, 1], cmap=cm.coolwarm, origin="lower", aspect="equal", vmin=0, vmax=1)


rhos = [1, 1.101, 1.183, 1.393]
for rho in rhos:
        ax.plot(x, rho*x, color="k")

x0  =.5
ax.axvline(x0, color=teal)

ax.set(title="$\Psi(\\rho)$", xlabel="$d_R$", ylabel="$d_R$", xlim=[0, 1], ylim=[0, 1])



#%%
f, ax = create_figure(subplots=False)

# get Psi and plot a slice of it
x = np.arange(0, 1, .01)
y = np.arange(0, 1, .01)

l = len(x)-1
Z = np.zeros((len(x), len(x)))

for i, xx in enumerate(x):
    for ii, yy in enumerate(y):
        psi = yy/xx
        Z[ii, i] = sigmoid(psi, L=.7, x0=1, k=15, b=.15)
ax.plot(x, Z[:,  50], color=orange)



# get the scled lengths of the L paths
x0 = .5
pathlengths = pa.paths_lengths.distance.values
alpha = pathlengths[-1] / x0
pathlengths = pathlengths / alpha
for y in pathlengths:
        ax.axvline(y, color="k", alpha=.5)

# get the real p(R)
prs = [.5, .75, .78, .85]
for pr in prs:
        ax.axhline(pr, color="k", alpha=.5)


# get P(R) axccording to Psi
rhos = [1, 1.101, 1.183, 1.393]
for rho in rhos:
        y = x0*rho
        pr = Z[int(y*100), 50]
        ax.scatter(y, pr, color=orange, s=20, zorder=20)
        print(rho, y, pr)


ax.set(xlabel="$d_L$", ylabel="$p(R)$")


#%%
