import pandas as pd
import matplotlib
import sys
sys.path.append('./')

import os
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import cv2
from collections import namedtuple

from Processing.plot.plotting_utils import get_bps_as_points_dict

def run(main, side, video):
    # Plotting stuff
    plt.ion()
    colors = dict(
        snout=[.8, .5, .1],
        left_ear = [.8, .1, .1],
        right_ear=[.5, .7, .1],
        body = [.6, .1, .6],
        tail_base = [.2, .8, .6],
        tail_2 = [.2, .5, .8],
        tail_3 = [.1, .3, .6]
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get pose data
    names = ['main', 'top', 'side']
    files = {}
    files['main'] = os.path.join(main, [f for f in os.listdir(main) if video in f and 'h5' in f][0])
    files['top'] = os.path.join(side, [f for f in os.listdir(side) if video in f and 'h5' in f and 'top' in f][0])
    files['side'] = os.path.join(side, [f for f in os.listdir(side) if video in f and 'h5' in f and 'side' in f][0])

    data_frames = {n:pd.read_hdf(files[n]) for n,f in files.items()}
    num_frames = len(data_frames['main'])
    print('Got data')

    # Get video readers
    video_files = {}
    video_files['main'] = os.path.join(main, [f for f in os.listdir(main) if video in f and 'mp4' in f and 'labeled' in f][0])
    video_files['top'] = os.path.join(side, [f for f in os.listdir(side) if video in f and 'mp4' in f and 'top' in f and 'labeled' in f][0])
    video_files['side'] = os.path.join(side, [f for f in os.listdir(side) if video in f and 'mp4' in f and 'side' in f and 'labeled' in f][0])

    readers = {n:cv2.VideoCapture(f) for n,f in video_files.items()}

    # Loop over frames
    for framen in range(num_frames):
        if framen < 125: continue
        print(framen)

        # Set plotting stuff
        ax.cla()
        # ax.set(xlim=[400, 900], ylim=[150, 350], zlim=[20, 200])

        # show cap frames
        for n, cap in readers.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, framen)
            ret, frame = cap.read()
            cv2.imshow(n, frame)

        # Get frame pose data as array
        poses = {n:df.iloc[framen] for n,df in data_frames.items()}
        points = {n: get_bps_as_points_dict(pose) for n,pose in poses.items()}
        bodyparts = points['main'].keys()

        # Create 3d scatter plot
        bp_to_plot = [ 'snout', 'left_ear', 'right_ear', 'body', 'tail_base', 'tail_2', 'tail_3']
        for bp in bodyparts:
            if bp not in bp_to_plot: continue

            ax.scatter(points['main'][bp][1], points['main'][bp][0], -points['top'][bp][1],
                       color=colors[bp], s=1000)

        segments = [('snout', 'right_ear'), ('snout', 'left_ear'), ('body', 'tail_base'), ('right_ear', 'body'),
                    ('left_ear', 'body'), ('tail_base', 'tail_2'), ('tail_2', 'tail_3')]
        for segment in segments:
            ax.plot([points['main'][segment[0]][1], points['main'][segment[1]][1]],
                    [points['main'][segment[0]][0], points['main'][segment[1]][0]],
                    [-points['top'][segment[0]][1], -points['top'][segment[1]][1]], 
                    color=colors[segment[0]], linewidth=20)

        plt.pause(0.5)
        # cv2.waitKey(0)

    while True:
        plt.pause(0.0001)


        


if __name__ == "__main__":
    main_fld = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\3dcam_test'
    side_fld = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\3dcam_test\\side'
    vide_name = 'vid_17'
    run(main_fld, side_fld, vide_name)



