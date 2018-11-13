from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from warnings import warn

from Processing.plot.plotting_utils import *


def plot_pose_timelapse(pose, stimframe, bp_colors, segments_colors, segments_names, 
                        ax=None, x_as_time=False, mark_each=5, start=0, stop=-1, savepath=None, stim_dur=None):
    '''plot_pose_timelapse [plots a "timelapse" of the pose of the mouse at each frame in a given range. The plot ca either be in the real X,Y
    position or with the X axis as time (x_as_time)]
    
    Arguments:
        pose {[pd.Dataframe]} -- [Pandas dataframe of the pose]
        stimframe {[int]} -- [frame at which the stimulus]
        bp_colors {[dict]} -- [dictionary of colors for the bodyparts: bp_name:bp_color]
        segments_colors {[dict]} -- [dictionary of colors for the body segments: segment_name:segment_color]
        segments_names {[dict]} -- [dictionary of tuples: segment_name:(bp1_name, bp2_name)]
    
    Keyword Arguments:
        ax {[ax]} -- [plt.axis] (default: {None})
        x_as_time {bool} -- [use X axis to show time insted of X position] (default: {False})
        mark_each {int} -- [mark with a vertical line and thicker plots every N frame] (default: {5})
        start {int} -- [frame number from which to start displaying poses] (default: {0})
        stop {int} -- [frame number from which to stop displaying poses] (default: {-1})
        savepath {[str]} -- [string to path where to save the folder (including name and extension)] (default: {None})
        stim_dur {[int]} -- [duration of the stimulus in frames] (default: {None})

    Raises:
        ValueError -- [description]
    '''

    if ax is None:
        f, ax = create_figure()
    ax.set(facecolor=[.2, .2, .2])

    # Get range of frames to process
    if stop == -1:
        stop = len(pose)
    poses_range = range(start, stop)

    # If adjusting X position - get x-axis range to plot over
    if x_as_time:
        if isinstance(x_as_time, bool): 
            x_as_time = 1000
            warn('x_as_time wasnt an integer, set to 1000')
        positions = np.linspace(0, x_as_time, len(poses_range))

    # Loop over frames and plot
    for i, framen in enumerate(poses_range):
        # change plotting params depending on frame number
        if framen == stimframe:
            size = 120
            lwidth = 10
            alpha = 1
            if x_as_time: ax.axvline(positions[i], color='r', linewidth=3, alpha=.75)
        elif framen % mark_each == 0:
            size = 120
            lwidth = 10
            alpha = 1
            if x_as_time: ax.axvline(positions[i], color='w', linewidth=1, alpha=.75)
        else:
            size = 25
            lwidth = 3
            alpha = .07
        if stim_dur is not None:
            if framen == stimframe + stim_dur:
                size = 120
                lwidth = 10
                alpha = 1
                if x_as_time:
                    ax.axvline(positions[i], color='r', linewidth=3, alpha=.75)

        # get bps position
        frame_pose = poses.iloc[framen]
        points_dict = get_bps_as_points_dict(frame_pose)
        if x_as_time:
            spacer = frame_pose.body.x - positions[i]  # ? only used if x_as_time is True
        else:
            spacer = 0

        # Plot bodyparts
        for name, (x, y) in points_dict.items():
            ax.scatter(bp.x-spacer, bp.y, color=bp_colors[name], s=40, alpha=alpha)

       # Plot body segments 
        for segment, (bp1, bp2) in segments_names.items():
            x.plot([pointsdict[bp1][0]-spacer, pointsdict[bp2][0]-spacer],
                   [pointsdict[bp1][1], pointsdict[bp2][1]], color=segments_colors[segment], linewidth=lwidth, alpha=alpha)

        # Save or display
        if savepath is not None:
            try:
                f.savefig(savepath, format=os.path.splitext(savepath)[-1], dpi=1000)
            except:
                raise ValueError('Could not save figure')
        else:
            show()






