import matplotlib.pyplot as plt
import os
import numpy as np
from Utilities.maths.math_utils import find_nearest

def get_bps_as_points_dict(frame_pose):
    '''get_bps_as_points_dict [turns the pose pd.Series into a dictionary, easier to handle]
    
    Arguments:
        frame_pose {[pd.Series]} -- [pose at one frame from DLC]
    
    Returns:
        [dict] -- [dictionary of x,y position of each bodypart in the frame]
    '''
    names = []
    pointsdict = {}
    bodyparts = frame_pose.index.levels[1]
    scorer = frame_pose.index.levels[0]
    for bpname in bodyparts:
        if bpname in names:  # dont take twice
            continue
        else:
            names.append(bpname)
            bp_pos = frame_pose[scorer[0], bpname]
            pointsdict[bpname] = np.int32([bp_pos.x, bp_pos.y])
    return pointsdict

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

def vline_to_curve(ax, x, xdata, ydata, **kwargs):
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
    ax.plot([x, x], [0, yval], **kwargs)

def hline_to_curve(ax, y, curve, **kwargs):
    """[plots a vertical line from the x axis to the curve at location x]
    
    Arguments:
        ax {[axarray]} -- [ax to plot on]
        y {[float]} -- [y value to plot on ]
        curve {[np.array]} -- [array of data with the curve. The horizzontal line will go from y on the y axis to the first point in 
                            which curve == y]
    """
    # try:
    #     x_stop = np.where(curve == y)[0][0]
    # except:
    #     x_stop = len(curve)

    # ax.plot([0, x_stop], [y, y], **kwargs)    
    raise NotImplementedError("need to make it work as for vline")