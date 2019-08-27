# %%
import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project
from Utilities.imports import *

from Processing.psychometric_analysis import PsychometricAnalyser
pa = PsychometricAnalyser()

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

%matplotlib inline


# %%
# set seaborn
import seaborn as sns
sns.set()
sns.set_style("white", {
            "axes.grid":"False",
            "ytick.right":"False",
            "ytick.left":"True",
            "xtick.bottom":"True",
            "text.color": "0"
})
mpl.rc('text', usetex=True)

# %%mat
# ! Psi parameters
s_ = 12
L_ = .6
b_ = (1 - L_)/2


# %%
sns.set_context("talk", font_scale=3)

# ? Make figure for upgrade
f, axarr = create_figure(subplots=False)

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

                z[i, ii] = p

img = axarr.imshow(z, alpha=1, aspect="equal", origin="lower", extent=[0, 1, 0, 1], cmap="coolwarm")
f.colorbar(img)

for r in [0.5, .2, 1, 1.5, 4.5]:
    axarr.plot(x, x*r, color="k", lw=4)

axarr.set(title="$\Psi$", xlabel="$d_R$", ylabel="$d_L$", xlim=[0, 1], ylim=[0, 1], xticks=[0, 1], yticks=[0, 1])
sns.despine(fig=f, offset=10, trim=False, left=False, right=True)


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
        psi = yy/xx
        PSI[ii, i] = psi
        Z[ii, i] = sigmoid(psi, L=L_, x0=1, k=s_, b=b_)


surf = ax.imshow(Z, extent=[0, 1, 0, 1], cmap=cm.coolwarm, origin="lower", aspect="equal", vmin=0, vmax=1)
f.colorbar(surf)

_x = [.2, .5, .8]

ax.set(title="$\Psi(\\rho)$", xlabel="$d_R$", ylabel="$d_R$")
sns.despine(offset=10, trim=False, left=False, right=True)

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




# %%
# Scale maze sizes to get points in utility space

# get the scled lengths of the L paths
x0 = .25
pathlengths = pa.paths_lengths.distance.values
alpha = pathlengths[-1] / x0
pathlengths = pathlengths / alpha


colors = [teal, green, purple, orange]
rhos = [round(y/x0, 2) for y in pathlengths]
prs = [0.78, 0.72, 0.70, 0.47]


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
        Z[ii, i] = sigmoid(psi, L=L_, x0=1, k=s_, b=b_)


surf = ax.imshow(Z, extent=[0, 1, 0, 1], cmap=cm.coolwarm, origin="lower", aspect="equal", vmin=0, vmax=1, alpha=.8)

for i, y in enumerate(pathlengths):
    ax.scatter(x0, y, s=80, color=colors[i], zorder=40, label="maze {}".format(i))
    ax.plot(x, rhos[i]*x, color=colors[i])

ax.legend()
ax.set(title="$\Psi(\\rho)$", xlabel="$d_R$", ylabel="$d_R$", xlim=[0, 1], ylim=[0, 1])



#%%
#  Fit Psy to the data
f, ax = create_figure(subplots=False)


for s in [7, 12]:

    # get Psi and plot a slice of it
    x = np.arange(0, 1, .01)
    y = np.arange(0, 1, .01)

    l = len(x)-1
    Z = np.zeros((len(x), len(x)))

    for i, xx in enumerate(x):
        for ii, yy in enumerate(y):
            psi = yy/xx
            Z[ii, i] = sigmoid(psi, L=L_, x0=1, k=s_, b=b_)
    ax.plot(x, Z[:,  25], color=black)




# get the real p(R) and dl
for pr, y, c in zip(prs, pathlengths, colors):
    # ax.axhline(pr, color=c, alpha=.5)
    # ax.axvline(y, color=c, alpha=.5)
    vline_to_point(ax, y, pr, color=c, ls="--", lw=2, alpha=.4)
    hline_to_point(ax, y, pr, color=c, ls="--", lw=2, alpha=.4)
    ax.scatter(y, pr, s=20, color=c)



ax.set(xlabel="$d_L$", ylabel="$p(R)$", xlim=[0, 1], ylim=[0, 1])


#%%


#%%
