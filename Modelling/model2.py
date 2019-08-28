# %%
import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project
from Utilities.imports import *

from Processing.psychometric_analysis import PsychometricAnalyser
pa = PsychometricAnalyser()

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
mpl.rc('text', usetex=False)
sns.set_context("talk", font_scale=1)  # was 3 

params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 150,  # to adjust notebook inline plot size
    'axes.labelsize': 8, # fontsize for x and y labels (was 10)
    'axes.titlesize': 8,
    'font.size': 8, # was 10
    'legend.fontsize': 6, # was 10
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'text.usetex': False,
    'figure.figsize': [3.39, 2.10],
}
matplotlib.rcParams.update(params)

# TODO look at slope relationship with the function

# %%
class PsiCalculator:
    colors = [lightblue, green, purple, magenta]

    good_fit_params = ([0.6, 0.8, 0],[0.9, 10, 2])

    def __init__(self):
        self.R = np.arange(0, 1010, 10)
        self.L = np.arange(0, 1010, 10)
        self.Psi = np.zeros((len(self.R), len(self.R)))
        self.PsidL, self.PsidR = np.zeros_like(self.Psi), np.zeros_like(self.Psi)

        # ? Psi params
        self.o = 1
        self.r0 = 1
        self.s = 8

        self.update_params()

        self.npoints = len(self.R)
        self.rmax = np.int(np.max(self.R))

        # Experimental data
        self.prs = [0.78, 0.72, 0.70, 0.47]
        # yerr: [2* math.sqrt(s) for s in list(sigmasquared.values())]
        self.yerrs = [0.048042182449778245, 0.0689335245891631, 0.06165147992851076, 0.07914616013894715]
        self.pathlengths = pa.paths_lengths.distance.values
        self.R0 = np.min(self.pathlengths)
        self.rhos = [round(y/self.R0, 2) for y in self.pathlengths]
        self._pathlengths = pa.paths_lengths.distance.values  # use a copy for scaling
        self._rhos = [round(y/self.R0, 2) for y in self.pathlengths]


    def update_params(self):
        self.default_params = {
            "o":self.o, 
            "r0":self.r0,
            "s":self.s
        }

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

    def calc_Psi_slice(self, l, o=1, s=10, r0=1, ):
        """
            Calcs the value of Psi along the vertical slice corresponiding to r0
        """ 
        return self.calc_Psi(l, self.R0, o=o, r0=r0, s=s)

    @staticmethod
    def dPsy_dL(l, r, o=1, r0=1, s=10):
        # returns the partial derivative of Psi over L at l,r
        e = np.exp(-(s*(l-r)/r))
        return (o*s*e)/((r*(1 + e)**2))

    @staticmethod
    def dPsy_dR(l, r, o=1, r0=1, s=10):
        # returns the partial derivative of Psi over R at l,r
        e1 = np.exp(-(s*(l-r)/r))
        e2 = np.exp(-((l-r)/r))
        return -((o*l*s*e1)/((r**2)*(1+e2)**2))

    def getPsi(self):
        for i, r in enumerate(self.R):
            for ii, l in enumerate(self.L):
                self.Psi[ii, i] = self.calc_Psi(l, r, **self.default_params)
                self.PsidL[ii, i] = self.dPsy_dL(l, r, **self.default_params)
                self.PsidR[ii, i] = self.dPsy_dR(l, r, **self.default_params)

    def get_arr_idx(self, x,):
        # get the index in the value closest to x
        return np.int(x*self.npoints / self.rmax)

    def plot_Psi(self, calc=False, ax=None, f=None, cbar=True):
        if calc: self.getPsi()

        if ax is None: f, ax = create_figure(subplots=False)

        surf = ax.imshow(self.Psi, extent=[0, self.rmax, 0, self.rmax], cmap=cm.coolwarm, origin="lower", aspect="equal")
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)

        if cbar: 
            f.colorbar(surf, cax=cax1)

        ax.set(title="$\Psi(\\rho)$", xlabel="$R$", ylabel="$L$", xticks=[0, self.rmax], yticks=[0, self.rmax])
        sns.despine(offset=10, trim=False, left=False, right=True)
        return ax

    def plot_Psy_derivs(self, calc=False, skip=0):
        if calc: self.getPsi()

        self.Psi, self.PsidL, self.PsidR = self.Psi[skip:, skip:], self.PsidL[skip:, skip:], self.PsidR[skip:, skip:]

        f, axarr = create_figure(subplots=True, ncols=3, nrows=2)
        surf = axarr[0].imshow(self.Psi, extent=[0, self.rmax, 0, self.rmax], cmap=cm.coolwarm, origin="lower", aspect="equal", vmin=0, vmax=1)
        surf = axarr[1].imshow(self.PsidL, extent=[0, self.rmax, 0, self.rmax], cmap=cm.coolwarm, origin="lower", aspect="equal", vmin=0, vmax=.005)
        surf = axarr[2].imshow(self.PsidR, extent=[0, self.rmax, 0, self.rmax], cmap=cm.coolwarm, origin="lower", aspect="equal", vmax=0, vmin=-.005)

        sns.despine(fig=f, offset=10, trim=False, left=False, right=True)

        for ax in axarr[:3]:
            # ax.axvline(self.R0, color=black, lw=2, ls="--")
            ax.set(xlabel="L", ylabel="Val", yticks=[])

        for r0 in np.arange(0, self.rmax, 100):
            r0idx = self.get_arr_idx(r0)
            axarr[1].axvline(r0, color=black, lw=1, ls="--")
            axarr[2].axvline(r0, color=black, lw=1, ls="--")
            axarr[0].axvline(r0, color=black, lw=1, ls="--")

            axarr[3].plot(self.Psi[:, r0idx], color=black, lw=2)
            axarr[4].plot(self.PsidL[:, r0idx], color=black, lw=2)
            axarr[5].plot(self.PsidR[:, r0idx], color=black, lw=2)

        titles = ["Psi", "partial L", "partial R"]
        for ax, t in zip(axarr, titles):
            ax.set(title=t, xticks=[], yticks=[], xlabel="R", ylabel="L")

    def plot_Psi_schematic(self):
        # ? Make figure for upgrade
        f, ax = create_figure(subplots=False)

        z = np.zeros_like(self.Psi)
        for i, xx in enumerate(self.R):
                for ii, yy in enumerate(self.L):
                        if xx == yy:
                                p = .5
                        elif xx > yy:
                                p = 1
                        else:
                                p = 0
                        z[i, ii] = p

        img = ax.imshow(z, alpha=1, aspect="equal", origin="lower", extent=[0, self.rmax, 0, self.rmax], cmap="coolwarm", vmin=0, vmax=1)
        f.colorbar(img)

        # for r in [0.5, .2, 1, 1.5, 4.5]:
        #     ax.plot(x, x*r, color="k", lw=4)

        ax.set(title="$\Psi(\\rho)$", xlabel="$R$", ylabel="$L$", xticks=[0, self.rmax], yticks=[0, self.rmax])
        sns.despine(fig=f, offset=10, trim=False, left=False, right=True)

    def plot_mazes_IC(self, ax=None, f=None, cbar=True):
        if ax is None: 
            ax = self.plot_Psi()
        else:
            self.plot_Psi(ax=ax, f=f, cbar=cbar)
        vline_to_point(ax, self.R0, np.max(self.pathlengths), color=black, ls="--")

        for L,c in zip(self.pathlengths, self.colors):
            ax.plot([0, self.rmax], [0, self.rmax*(L/self.R0)], color=c, lw=3)
            ax.scatter(self.R0, L, color=c, s=120, edgecolors=black, zorder=20)
        ax.set(title="$\Psi(\\rho)$", xlabel="$R$", ylabel="$L$", xticks=[0,  self.R0, self.rmax], yticks=[0, self.rmax], xlim=[0, self.rmax], ylim=[0, self.rmax])
        return ax

    def plot_mazes(self, ax=None, f=None, cbar=True, calc=False):
        if ax is None: 
            ax = self.plot_Psi()
        else:
            self.plot_Psi(ax=ax, f=f, cbar=cbar, calc=calc)

        vline_to_point(ax, self.R0, np.max(self.pathlengths), color=black, ls="--")

        yticks = [0]
        yticks.extend(list(self.pathlengths))
        yticks.extend([self.rmax])

        for L,c in zip(self.pathlengths, self.colors):
            hline_to_point(ax, self.R0, L, color=c, ls="--")
            ax.scatter(self.R0, L, color=c, s=120, edgecolors=black, zorder=20)
        ax.set(title="$\Psi(\\rho)$", xlabel="$R$", ylabel="$L$", xticks=[0,  self.R0, self.rmax], 
        yticks=yticks, xlim=[0, self.rmax], ylim=[0, self.rmax])
        return ax

    def fit_plot(self, fit_bounds=None):
        if fit_bounds is None:
            fit_bounds = self.good_fit_params
        # Plot the fit of the Psi function to the data
        f, axarr = create_figure(subplots=True, ncols=2)
        

        for L, c, pr, yerr in zip(self.pathlengths, self.colors, self.prs, self.yerrs):
            axarr[0].scatter(L, pr, color=c, s=120, edgecolors=black, zorder=20)
            hline_to_point(axarr[0], L, pr, color=c, lw=2, ls="--")

        params = plot_fitted_curve(self.calc_Psi_slice, self.pathlengths, self.prs, axarr[0], 
            xrange=[0, self.rmax],
            fit_kwargs={"sigma":self.yerrs, "method":"dogbox", "bounds":fit_bounds},
            scatter_kwargs={"alpha":0}, 
            line_kwargs={"color":black, "alpha":1, "lw":2,})

        axarr[0].set(title="Fit", xlim=[0, self.rmax], ylim=[0, 1], xlabel="L", ylabel="p(R)")

        print(""" 
            Fitted sigmoid:
                omega: {}
                slope: {}
                x0:    {}
        
        """.format(round(params[0], 2), round(params[1], 2), round(params[2], 2)))

        self.o = params[0]
        self.r0 = params[2]
        self.s = params[1]
        self.update_params()

        # Plot image and  set axes dimensions
        ax = self.plot_mazes(ax=axarr[1], f=f, cbar=True, calc=True)

        gs = GridSpec(1, 3, width_ratios=[1, 1, .25], height_ratios=[1])
        for i in range(len(axarr)):
            axarr[i].set_position(gs[i].get_position(f))

# %% test
calc = PsiCalculator()
calc.getPsi()
# params = calc.fit_plot(fit_bounds=([0.6, 0.8, 0],[0.9, 10, 2]))
calc.plot_Psy_derivs()




#%%


#%%
