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
mpl.rc('text', usetex=False)
sns.set_context("talk", font_scale=1)  # was 3 


# %%
class PsiCalculator:
    colors = [teal, green, purple, orange]

    def __init__(self):
        self.R = np.arange(0, 1010, 10)
        self.L = np.arange(0, 1010, 10)
        self.Psi = np.zeros((len(self.R), len(self.R)))
        self.PsidL, self.PsidR = np.zeros_like(self.Psi), np.zeros_like(self.Psi)

        # ? Psi params
        self.o = 1
        self.r0 = 1
        self.s = 10

        self.default_params = {
            "o":self.o, 
            "r0":self.r0,
            "s":self.s
        }

        self.npoints = len(self.R)
        self.rmax = np.int(np.max(self.R))


        # Experimental data
        self.prs = [0.78, 0.72, 0.70, 0.47]
        # yerr: [2* math.sqrt(s) for s in list(sigmasquared.values())]
        self.yerr = [0.048042182449778245, 0.0689335245891631, 0.06165147992851076, 0.07914616013894715]
        self.pathlengths = pa.paths_lengths.distance.values
        self.R0 = np.min(self.pathlengths)
        self.rhos = [round(y/self.R0, 2) for y in self.pathlengths]
        self._pathlengths = pa.paths_lengths.distance.values  # use a copy for scaling
        self._rhos = [round(y/self.R0, 2) for y in self.pathlengths]


    def scale_pathlengths(self, r0):
        # Scale path length so that the right arm has length r0
        alpha = self._pathlengths[-1] / r0
        self.pathlengths = self._pathlengths / alpha


        self.rhos = [round(y/r0, 2) for y in self.pathlengths]

    @staticmethod
    def calc_Psi(l, r, o=1, r0=1, s=10):
        """
        2D logistic
        l, r = values
        o = omega, scales the function in the Z direction
        r0 = value of l/r at which Psi=0.5
        s = slope
        """ 
        b = (1 - o)/2  # -> this ensures that the logistic is always centered on z=.5
        rho = l/r
        delta_rho = rho - r0
        z = o / (1 + np.exp(-s*(delta_rho)))+b
        return z

    @staticmethod
    def dPsy_dL(l, r, o=1, r0=1, s=10):
        # returns the partial derivative of Psi over L at l,r
        e = np.exp(-(s*(l-r)/r))
        return (o*s*e)/((r*(1 + e)**2))

    @staticmethod
    def dPsy_dR(l, r, o=1, r0=1, s=10):
        # returns the partial derivative of Psi over R at l,r
        e = np.exp(-(s*(l-r)/r))
        return -((o*l*s*e)/((r**2)*(1+e)**2))

    def getPsi(self):
        for i, r in enumerate(self.R):
            for ii, l in enumerate(self.L):
                self.Psi[ii, i] = self.calc_Psi(l, r, **self.default_params)
                self.PsidL[ii, i] = self.dPsy_dL(l, r, **self.default_params)
                self.PsidR[ii, i] = self.dPsy_dR(l, r, **self.default_params)

    def plot_Psy(self, calc=False, ax=None):
        if calc: self.getPsi()

        if ax is None: f, ax = create_figure(subplots=False)
        surf = ax.imshow(self.Psi, extent=[0, self.rmax, 0, self.rmax], cmap=cm.coolwarm, origin="lower", aspect="equal", vmin=0, vmax=1)
        f.colorbar(surf)
        ax.set(title="$\Psi(\\rho)$", xlabel="$R$", ylabel="$L$", xticks=[0, self.rmax], yticks=[0, self.rmax])
        sns.despine(offset=10, trim=False, left=False, right=True)
        return ax

    def plot_Psy_derivs(self, calc=False):
        if calc: self.getPsi()

        f, axarr = create_figure(subplots=True, ncols=3, sharex=True, sharey=True)
        surf = axarr[0].imshow(self.Psi, extent=[0, self.rmax, 0, self.rmax], cmap=cm.coolwarm, origin="lower", aspect="equal", vmin=0, vmax=1)
        surf = axarr[1].imshow(self.PsidL, extent=[0, self.rmax, 0, self.rmax], cmap=cm.coolwarm, origin="lower", aspect="equal")
        surf = axarr[2].imshow(self.PsidR, extent=[0, self.rmax, 0, self.rmax], cmap=cm.coolwarm, origin="lower", aspect="equal")

        sns.despine(fig=f, offset=10, trim=False, left=False, right=True)

        titles = ["Psi", "partial L", "partial R"]
        for ax, t in zip(axarr, titles):
            ax.set(title=t, xticks=[], yticks=[], xlabel="R", ylabel="R")


    def plot_mazes_IC(self):
        ax = self.plot_Psy()
        ax.axvline(self.R0)


# %% test
calc = PsiCalculator()
calc.getPsi()
calc.plot_mazes_IC()

# %%

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
        s = yy**2
        # Z[ii, i] = sigmoid(psi, L=L_, x0=1, k=s_, b=b_)
        Z[ii, i] = hill_function(psi, 15)


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
f, ax = create_figure(subplots=False)
f2, slopes_ax = create_figure(subplots=False)

slopes = []

prs = [0.78, 0.72, 0.70, 0.47]
# yerr: [2* math.sqrt(s) for s in list(sigmasquared.values())]
yerr = [0.048042182449778245, 0.0689335245891631, 0.06165147992851076, 0.07914616013894715]
xr = [0, 3]

# get the scled lengths of the L paths
x0s =  np.arange(0.5, 2, .1)
for x0 in  x0s:
    pathlengths = pa.paths_lengths.distance.values
    alpha = pathlengths[-1] / x0
    pathlengths = pathlengths / alpha


    colors = [teal, green, purple, orange]
    rhos = [round(y/x0, 2) for y in pathlengths]

    x = np.linspace(xr[0], xr[1], 101)
    # ? Plot sigmoid filled to psy - mean pR of grouped bayes + std
    pomp = plot_fitted_curve(centered_logistic, pathlengths, prs, ax, 
        xrange=xr,
        fit_kwargs={"sigma":yerr, "method":"dogbox", "bounds":([0.0, 0, 5,],[2.21, 2, 38])},
        scatter_kwargs={"alpha":0}, 
        line_kwargs={"color":black, "alpha":.85, "lw":2,})

    slopes.append(pomp[-1])

    # get the real p(R) and dl
    # for pr, y, c in zip(prs, pathlengths, colors):
    #     # ax.axhline(pr, color=c, alpha=.5)
    #     # ax.axvline(y, color=c, alpha=.5)
    #     vline_to_point(ax, y, pr, color=c, ls="--", lw=2, alpha=.4)
    #     hline_to_point(ax, y, pr, color=c, ls="--", lw=2, alpha=.4)
    #     ax.scatter(y, pr, s=20, color=c)



ax.set(xlabel="$d_L$", ylabel="$p(R)$", xlim=xr, ylim=[0, 1])


X, Y = np.log(x0s), np.log(slopes)
slopes_ax.scatter(X, Y)

slopes_fit = polyfit(1, X, Y)
predicted = slopes_fit(X)
# X, p0, p1, fit = linear_regression(X, Y)
# predicted = [p0 + p1*x for x in X]
slopes_ax.plot(X, predicted, color=black)

slopes_ax.set(xlabel="$d_R", ylabel="slope")
# %%
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
# Playing with sigmoids

funcs = [logistic,arctangent,error_function]
func_names = ["logistic", "arctangent",  "error_function"]
func_params = {
    "logistic":[1, 1, 1.25, 0],

    "arctangent": [3, 0.5, 1],

    "error_function": [1, 1.25, 1, 0],
}

f, ax = create_figure(subplots=False)

for fn, f in zip(func_names, funcs):
    x = np.arange(-3, 4, .01)

    if fn != "algebraic_sigmoid":
        y = f(x, *func_params[fn])
    else:
        y = [f(xx, *func_params[fn]) for xx in x]
    ax.plot(x, y, label=fn)

ax.legend()



#%%
f, ax = plt.subplots()
plt.scatter([0.5, 0.75, 1.5, 2], [37, 24.6, 12, 8.9])


#%%
