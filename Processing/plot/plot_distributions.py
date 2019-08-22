import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project

from scipy import stats
import numpy as np
from scipy.optimize import curve_fit

from Processing.plot.plotting_utils import *
from Utilities.matplotlib_config import *

def plot_distribution(*args, dist_type="logistic", comulative=False, ax=None, shaded=False, x_range=None, plot_kwargs={}, ax_kwargs={},  **kwargs):
    # Get the distribution
    if dist_type == "logistic":
        dist = stats.logistic(*args, **kwargs)
    elif dist_type == "gamma":
        dist = stats.gamma(*args, **kwargs)
    elif dist_type == "beta":
        dist = stats.beta(*args, **kwargs)
    elif dist_type == "exponential":
        dist = np.exp
    else:
        raise NotImplementedError

    # Get the probability density function or comulative density function
    if comulative: func = dist.cdf
    else: 
        try:
            func = dist.pdf
            x = np.linspace(dist.ppf(0.0001), dist.ppf(0.99999), 100)
        except:
            func = dist
            x = np.linspace(x_range[0], x_range[1], 100)

    # Plot
    if ax is None: f, ax = plt.subplots()

    if not shaded:
        ax.plot(x, func(x), **plot_kwargs)
    else: 
        ax.fill_between(x, 0, func(x), **plot_kwargs)
        ax.plot(x, func(x), **plot_kwargs)

    ax.set(**ax_kwargs)

    return ax

def plot_fitted_curve(func, xdata, ydata, ax, xrange=None, print_fit=False, numpy_polyfit=False,
                        fit_kwargs={}, scatter_kwargs={}, line_kwargs={}):
    if numpy_polyfit and not isinstance(numpy_polyfit, int): raise ValueError("numpy_polyfit should be an integer")
    # set numpy_polifit to an integer to fit a numpy polinomial with degree numpy polyfit

    if xrange is not None: x = np.linspace(xrange[0], xrange[1], 100)
    else: x = np.linspace(np.min(xdata), np.max(xdata), 100)

    if not numpy_polyfit: # ? scipy curve fit instead
        popt, pcov = curve_fit(func, xdata, ydata, **fit_kwargs)
        if print_fit: print(popt)
        y = func(x, *popt)
        to_return = popt
    else:
        func = func(numpy_polyfit, xdata, ydata)
        y = func(x)
        to_return = func

    ax.scatter(xdata, ydata, **scatter_kwargs)
    ax.plot(x, y, **line_kwargs)

    return to_return

def dist_plot(dist, ax=None, **kwargs):
    # ? this is differente from plot distribution because we already have the distribution, just want to plot it
    if ax is None: plt.subplots()
    
    x = np.linspace(dist.ppf(0.0001), dist.ppf(0.99999), 100)
    ax.plot(x, dist.pdf(x), **kwargs)
    
def plot_kde(ax, kde, z, invert=False, vertical=False, normto=None, label=None, **kwargs):
    """[Plots a KDE distribution. Plots first the shaded area and then the outline. 
       KDE can be oriented vertically, inverted, normalised...]
    
    Arguments:
        ax {[plt.axis]} -- [ax onto which to plot]
        kde {[type]} -- [KDE fitted with statsmodels]
        z {[type]} -- [value used to shift the curve, for a horizontal KDE z=0 means the curve is on the x axis. ]
    
    Keyword Arguments:
        invert {bool} -- [mirror the KDE plot relative to the X or Y axis, depending on ortentation] (default: {False})
        vertical {bool} -- [plot KDE vertically] (default: {False})
        normto {[float]} -- [normalise the KDE so that the peak of the distribution is at a certain value] (default: {None})
        label {[string]} -- [label for the legend] (default: {None})
    
    Returns:
        ax, kde
    """
    if vertical:
        x = kde.density
        y = kde.support
    else:
        x, y = kde.support, kde.density
    
    if normto is not None:
        if not vertical:
            y = y / np.max(y) * normto
        else:
            x = x / np.max(x) * normto

    if invert:
        y = z - y
    else: 
        if not vertical: y = y + z
        else: x = x + z

    plot_shaded_withline(ax, x, y, z, **kwargs)

    return ax, kde
        

if __name__ == "__main__":
    plot_distribution(1.00, 1.00, dist_type="beta", shaded="True", ax_kwargs={"ylim":[0, 1.1]}, plot_kwargs={"color":[.4, .8, .2]})
    plt.show()