import pandas as pd
import matplotlib
import sys
sys.path.append('./')

import os
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from collections import namedtuple

from Processing.plot.plotting_utils import get_bps_as_points_dict

def run():
    plt.ion()

    colors = dict(
        snout=[.6, .2, .2],
        left_ear = [.5, .3, .4],
        right_ear=[.5, .4, .3],
        body = [.2, .5, .8],
        tail_base = [.2, .8, .2],
        tail_2 = [.2, .5, .2]
    )


    fld = '/Users/federicoclaudi/Desktop'
    names = ['main', 'top', 'side']

    files = {}
    for name in names:
        files[name] = [f for f in os.listdir(fld) if 'h5' in f and name in f][0]

    data_frames = {n:pd.read_hdf(os.path.join(fld, f)) for n,f in files.items()}
    num_frames = len(data_frames['main'])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    bp_poses = namedtuple('poses', 'main top side')
    print('Got data')
    for framen in range(num_frames):
        if framen < 200: continue
        ax.cla()
        ax.set(xlim=[400, 900], ylim=[150, 350], zlim=[20, 200])

        print(framen)
        poses = {n:df.iloc[framen] for n,df in data_frames.items()}

        points = {n: get_bps_as_points_dict(pose) for n,pose in poses.items()}

        bodyparts = points['main'].keys()

        bp_to_plot = ['snout', 'left_ear', 'right_ear', 'body', 'tail_base', 'tail_2']
        for bp in bodyparts:
            if bp not in bp_to_plot: continue

            ax.scatter(points['main'][bp][0], points['main'][bp][1], points['top'][bp][1],
                       color=colors[bp], s=7000)
        plt.pause(0.0001)




if __name__ == "__main__":
    run()



