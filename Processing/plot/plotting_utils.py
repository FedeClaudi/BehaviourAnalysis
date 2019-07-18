import matplotlib.pyplot as plt
import os
import numpy as np


def get_bps_as_points_dict(frame_pose):
    '''get_bps_as_points_dict [turns the pose pd.Series into a dictionary, easier to handle]
    
    Arguments:
        frame_pose {[pd.Series]} -- [pose at one frame from DLC]
    
    Returns:
        [dict] -- [dictionary of x,y position of each bodypart in the frame]
    '''

    # TODO add other variables (e.g. velocity...)
    # TODO exclude/include body segments

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

def style_legend(ax):
    l = ax.legend()
    for text in l.get_texts():
        text.set_color([.7, .7, .7])

def ortholines(ax, orientations, values, color=[.7, .7, .7], lw=3, alpha=.5, ls="--",  **kwargs):
    if not isinstance(orientations, list): orientations = [orientations]
    if not isinstance(values, list): values = [values]

    for o,v in zip(orientations, values):
        if o == 0:
            func = ax.hline
        else:
            func = ax.vline

        func(v, color=[.7, .7, .7], lw=3, alpha=.5, ls="--", **kwargs)