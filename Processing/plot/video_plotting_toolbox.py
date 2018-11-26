import cv2
import pandas as pd
import numpy as np
import os
from collections import namedtuple
import matplotlib.pyplot as plt

from Processing.plot.plotting_utils import *


def cv2_plot_mouse_bps(frame, points_dict, color_dict=None, s=2):
    '''cv2_plot_mouse_bps [Plots each bodypart as a circle over the frame]
    
    Arguments:
        frame {[np.array]} -- [video frame]
        points_dict {[dict]} -- [created by get_bps_as_points_dict()]
    
    Keyword Arguments:
        color_dict {[dict]} -- [dictionary of color_name-color_value for each bodypart] (default: {None})
        s {int} -- [size of the circle being plotted] (default: {5})
    '''
    if color_dict is None: color_dict = {}

    for name, (x, y) in points_dict.items():
        # Get color
        if name in color_dict.keys():
            color = color_dict[name]
        else:
            color = [0, 0, 255]
        # Plot circle
        cv2.circle(frame, (np.int32(x),
                           np.int32(y)), s, color, -1)

def custom_plot_poly(frame, points_dict):
    def plot_segments(frame, points_dict, segments, color, lw):
        # cv2.line(image, (x1, y1), (x2, y2), (0,255,0), lineThickness)
        for seg in segments:
            bp1 = points_dict[seg.bp1]
            bp2 = points_dict[seg.bp2]

            cv2.line(frame, (bp1[0], bp1[1]), (bp2[0], bp2[1]), color, lw)

    segment = namedtuple('segment', 'bp1 bp2')
    head_segments = [segment('left_ear', 'snout'), segment('snout', 'right_ear'),
                     segment('right_ear','neck'), segment('neck', 'left_ear')]

    body_axis = [segment('neck', 'body'), segment('body', 'tail_base')]
    # body_contour = [segment('left_ear', 'left_shoulder'), segment('left_shoulder', 'left_hip'),
    #                 segment('left_hip', 'tail_base'), segment('tail_base', 'right_hip'),
    #                 segment('right_hip', 'right_should'), segment('right_should', 'right_ear')]
    tail = [segment('tail_base', 'tail_2'), segment('tail_2', 'tail_3')]

    contour_names = ['left_ear', 'snout', 'right_ear', 'right_should', 'right_hip', 'tail_base',
                     'left_hip', 'left_shoulder']

    colors = dict(
                  head = [255, 200, 200],
                  body_axis = [200, 200, 255],
                  body_contour=[50, 255, 128],
                  tail = [200, 150, 120])

    cv2.fillPoly(frame, [np.int32([points_dict[k] for k in contour_names])], color=colors['body_contour'])
    plot_segments(frame, points_dict, head_segments, colors['head'], 3)
    plot_segments(frame, points_dict, body_axis, colors['body_axis'], 3)
    plot_segments(frame, points_dict, tail, colors['tail'], 3)


def cv2_plot_mouse_poly(frame, points_dict, include_names=None, color=None, mode='fill'):
    if color is None: color = [0, 0, 0]
    if include_names is None: 
        include_names = list(points_dict.keys())
    else:
        if not isinstance(include_names, list): include_names = [include_names]

    if mode == 'fill':
        cv2.fillPoly(frame, [np.int32([points_dict[k] for k in include_names])],
                        color=colors.head)
    elif mode == 'lines':
        cv2.polylines(frame, [np.int32([points_dict[k] for k in include_names])],
                        color=color, isClosed=False)
    else:
        raise ValueError('Unrecognised plotting mode: ', mode)


def overlay_tracking_on_video(videopath=None, posepath=None, posedata=None, output_format = 'mp4', savepath=None,
                                blk_frame=False, plot_points=False, plot_poly=False, poly_mode='fill',
                                plot_custom=True, colors_dict=None, cap=None, cv_writer=None):
    '''overy_tracking_on_video [overlays plots of mouse pose over a video]
    
    Keyword Arguments:
        videopath {[str]} -- [path to video file] (default: {None})
        posepath {[str]} -- [path to pose data from DLC] (default: {None})
        posedata {[pd.Dataframe]} -- [pose data - in atlernative to posepath] (default: {None})
        output_format {str} -- [format of the video being output] (default: {'mp4'})
        savepath {[str]} -- [path where the video will be stored - NECESSARY] (default: {None})
        blk_frame {bool} -- [if true the video will have a black background instead of being overlaid over real video] (default: {False})
        plot_points {bool} -- [plot bp as circles] (default: {False})
        plot_poly {bool} -- [plot bps connected as polygon] (default: {True})
        poly_mode {str} -- [plot poly lines as lines or filled polygon] (default: {'fill'})
        colors_dict {[dict]} -- [dictionary bp_name:color] (default: {None})
        cap {[cv2.cap]} -- [videofilecapture - in alternative to videopath] (default: {None})
        cv_writer {[cv2.VideoWriter]} -- [videowriter] (default: {None})

    '''

    # Check correct arguments have been passed and get pose data 
    if posepath is None and posedata is None:
        raise ValueError('No pose data passed to the function')
    elif posepath is not None and posedata is None:
        if not isinstance(posepath, str):
            raise ValueError('Invalid arguments')
        if not 'h5' in posepath:
            raise ValueError('Invalid file format for pose data')
        pose = pd.read_hdf(posepath)  # ! <--
    elif posepath is None and posedata is not None:
        if not isinstance(posedata, pd.DataFrame):
            raise ValueError('Invalid data type for pose data')
        pose = posedata  # ! <--
    else:
        use_posedata = input('\n\nPassed both posepath and posedata, use posedata?  y/n  ')
        if use_posedata != 'y':
            pose = pd.read_hdf(posepath)  # ! <--
        else:
            if isinstance(posedata, pd.DataFrame):
                pose = posedata  # ! <--
            else: raise ValueError('Invalid data type for pose data')

    if not videopath is None:
        if not isinstance(videopath, str):
            raise ValueError('Invalid arguments')
        if not 'avi' and not 'mp4' in videopath:
            raise ValueError('Invalid video format - {} for video {}'.format(os.path.splitext(videopath)[-1], videopath))

        if not 'mp4' in output_format:
            raise ValueError('Format {} not implemented yet'.format(output_format))
    else:
        if cap is None:
            raise ValueError('Either videopath or cap arguments should be passed')

    if savepath is None: raise ValueError('No save path passed')

    if colors_dict is not None:
        if not isinstance(colors_dict, dict):
            raise ValueError('wrong fromat for colors dictionary')

    # Set up openCV file reader and writer if they were not passed to the function
    if cap is None:
        cap = cv2.VideoCapture(videopath)

    if cv_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv_writer = cv2.VideoWriter(savepath, fourcc, 30, (w, h), False)

    # Start looping over frames
    framen = 0  
    cap.set(cv2.CAP_PROP_POS_FRAMES, framen)  # ? start from the beginning
    while True:
        # Get frame and pose data
        if framen > 0 and framen % 100 == 0:
            print('frame n: ', framen)

        ret, frame = cap.read()
        if not ret:
            break

        if blk_frame:
            frame = np.zeros((frame.shape), np.uint8)

        frame_pose = pose.iloc[framen]
        points_dict = get_bps_as_points_dict(frame_pose)

       

        if plot_poly:
            cv2_plot_mouse_poly(frame, points_dict, include_names=None, mode=poly_mode)

        if plot_custom:
            custom_plot_poly(frame, points_dict, )

        if plot_points:
            cv2_plot_mouse_bps(frame, points_dict, s=3)

        cv_writer.write(frame)

        framen += 1
    cv_writer.release()







