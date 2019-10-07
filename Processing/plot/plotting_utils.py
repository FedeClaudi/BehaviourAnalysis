from Utilities.matplotlib_config import *
import os
import numpy as np
from Utilities.maths.math_utils import find_nearest
from matplotlib import cm
from matplotlib.colors import Normalize

# ! COLORS


class InvertedNormalize(Normalize):
    def __call__(self, *args, **kwargs):
        return 1 - super(InvertedNormalize, self).__call__(*args, **kwargs)

class MplColorHelper:
    """
        Usage: instantiate the class with the CMAP to be used and the coors range. Then pass it values to get the RGB value of the color.
        "inverse" gives the possibility to invert the order of the colors in the cmap
    """
    def __init__(self, cmap_name, start_val, stop_val, inverse=False):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)

        if not inverse:
            self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        else:
            self.norm = InvertedNormalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)

def get_n_colors(n, cmap="tab20"):
	return [plt.get_cmap(cmap)(i) for i in np.arange(n)]

def desaturate_color(c, k=.5):

    return sns.desaturate(c, k)

def saturate_color(c, k=.5):
    return sns.saturate(c, k)

# ! other stuf
def clean_axes(f):
    sns.despine(fig=f, offset=10, trim=False, left=False, right=True)

def save_all_open_figs(target_fld=False, name=False, format=False, exclude_number=False):
    open_figs = plt.get_fignums()

    for fnum in open_figs:
        if name:
            if not exclude_number: ttl = '{}_{}'.format(name, fnum)
            else: ttl = str(name)
        else:
            ttl = str(fnum)

        if target_fld: ttl = os.path.join(target_fld, ttl)
        if not format: ttl = '{}.{}'.format(ttl, 'svg')
        else: ttl = '{}.{}'.format(ttl, format)

        plt.figure(fnum)
        plt.savefig(ttl)


def create_figure(subplots=True, **kwargs):
    if "figsize" not in kwargs.keys():
        kwargs["figsize"] = (12, 8)
    if not subplots:
        f, ax = plt.subplots(**kwargs)
    else:
        f, ax = plt.subplots(**kwargs)
        ax = ax.flatten()
    return f, ax

def show(): plt.show()


def ticksrange(start, stop, step):
    return np.arange(start, stop + step, step)


def save_figure(f, path):
    f.savefig(path)

def close_figure(f):
    plt.close(f)

def make_legend(ax):
    l = ax.legend()
    for text in l.get_texts():
        text.set_color([.7, .7, .7])



# ! plotting functions
def ortholines(ax, orientations, values, color=[.7, .7, .7], lw=3, alpha=.5, ls="--",  **kwargs):
    """[makes a set of vertical and horizzontal lines]
    
    Arguments:
        ax {[np.axarr]} -- [ax]
        orientations {[int]} -- [list of 0 and 1 with the orientation of each line. 0 = horizzontal and 1 = vertical]
        values {[float]} -- [list of x or y values at which the lines should be drawn. Should be the same length as orientations]

    """
    if not isinstance(orientations, list): orientations = [orientations]
    if not isinstance(values, list): values = [values]

    for o,v in zip(orientations, values):
        if o == 0:
            func = ax.axhline
        else:
            func = ax.axvline

        func(v, color=color, lw=lw, alpha=alpha, ls=ls, **kwargs)

def vline_to_curve(ax, x, xdata, ydata, dot=False, line_kwargs={}, scatter_kwargs={}, **kwargs):
    """[plots a vertical line from the x axis to the curve at location x]
    
    Arguments:
        ax {[axarray]} -- [ax to plot on]
        x {[float]} -- [x value to plot on ]
        curve {[np.array]} -- [array of data with the curve. The vertical line will go from 0 to curve[x]]
    """
    line = ax.plot(xdata, ydata, alpha=0)
    xline, yline = line[0].get_data()
    x = find_nearest(xline, x)
    yval = yline[np.where(xline == x)[0][0]]
    ax.plot([x, x], [0, yval], **line_kwargs)
    if dot:
        ax.scatter(x, yval, **scatter_kwargs, **kwargs)

def vline_to_point(ax, x, y, **kwargs):
    ax.plot([x, x], [0, y], **kwargs)
def hline_to_point(ax, x, y, **kwargs):
    ax.plot([0, x], [y, y], **kwargs)

def hline_to_curve(ax, y, xdata, ydata, dot=False, line_kwargs={}, scatter_kwargs={}, **kwargs):
    """[plots a vertical line from the x axis to the curve at location x]
    
    Arguments:
        ax {[axarray]} -- [ax to plot on]
        x {[float]} -- [x value to plot on ]
        curve {[np.array]} -- [array of data with the curve. The vertical line will go from 0 to curve[x]]
    """
    line = ax.plot(xdata, ydata, alpha=0)
    xline, yline = line[0].get_data()
    y = find_nearest(yline, y)
    xval = xline[np.where(yline == y)[0][0]]
    ax.plot([0, xval], [y, y], **line_kwargs, **kwargs)
    if dot:
        ax.scatter(xval, y, **scatter_kwargs, **kwargs)

def plot_shaded_withline(ax, x, y, z=None, label=None, alpha=.15,  **kwargs):
    """[Plots a curve with shaded area and the line of the curve clearly visible]
    
    Arguments:
        ax {[type]} -- [matplotlib axis]
        x {[np.array, list]} -- [x data]
        y {[np.array, list]} -- [y data]
    
    Keyword Arguments:
        z {[type]} -- [description] (default: {None})
        label {[type]} -- [description] (default: {None})
        alpha {float} -- [description] (default: {.15})
    """
    if z is not None:
        # ax.fill_between(x, z, y, alpha=alpha, **kwargs)
        ax.fill_betweenx(y, z, x, alpha=alpha, **kwargs)

    else:
        # ax.fill_between(x, y, alpha=alpha, **kwargs)
        ax.fill_betweenx(y, x, alpha=alpha, **kwargs)

    ax.plot(x, y, alpha=1, label=label, **kwargs)