# %%
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# %maptlotlib inline
# TODO add possibility to shade area under the curve
# %%
def plot_distribution(*args, dist_type="logistic", comulative=False, xlim=[0, 10], ylim=[0, 1], 
                    color="r", alpha=1, title=None, ax=None, label=None,  **kwargs):
    # Get the distribution
    if dist_type == "logistic":
        dist = stats.logistic(*args, **kwargs)
    elif dist_type == "gamma":
        dist = stats.gamma(*args, **kwargs)
    elif dist_type == "beta":
        dist = stats.beta(*args, **kwargs)
    else:
        raise NotImplementedError

    # Get the probability density function or comulative density function
    if comulative: func = dist.cdf
    else: func = dist.pdf

    # Plot
    if ax is None: f, ax = plt.subplots()
    x = np.linspace(dist.ppf(0.0001), dist.ppf(0.99999), 100)
    ax.plot(x, func(x), color=color, lw=4, alpha=alpha, label=label)

    ax.set(title=title, xlim=xlim, ylim=ylim)

    return ax

def plot_fitted_sigmoid(func, xdata, ydata, xrange, ax, scatter_kwargs={}, line_kwargs={}):
    popt, pcov = curve_fit(func, xdata, ydata)

    x = np.linspace(xrange[0], xrange[1], 100)
    y = func(x, *popt)

    ax.scatter(xdata, ydata, **scatter_kwargs)
    ax.plot(x, y, **line_kwargs)




# %%
if __name__ == "__main__":
    # Plot some curves
    ax = plot_distribution(dist_type="logistic", comulative=True, loc=1, scale=.5, xlim=[0, 2], alpha=.25)
    ax = plot_distribution(dist_type="logistic", comulative=True, loc=1, scale=2, xlim=[0, 2], alpha=1, ax=ax)

    ax.axvline(1.76, ls=":", color="w", alpha=.4, lw=1)
    ax.axhline(0.84, ls=":", color="w", alpha=.4, lw=1)

    ax.axvline(1, ls=":", color="w", alpha=.4, lw=1)
    ax.axhline(0.5, ls=":", color="w", alpha=.4, lw=1)

    #%%
