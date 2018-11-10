from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2


dr = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\DAQ\\upstairs_rig\\video_clips\\videos_for_FC'
videoname = "Barnes US wall up (2)_CA3380_audio-3 (27')"
posename = "Barnes US wall up (2)_CA3380_audio-3 (27')DeepCut_resnet50_BarnesNov9shuffle1_150000_labeled"

# videos = [f for f in os.listdir(dr) if'avi' in f and 'labeled' not in f]
# poses = [f for f in os.listdir(dr) if 'h5' in f and 'metadata' not in f]
videos = [os.path.join(dr, videoname)]
poses = [os.path.join(dr, posename)]

cols = namedtuple('colors', 'head body')
colors = cols([.8, .4, .4], [.4, .8, .8])
bp_groups = cols(['snout', 'rear', 'lear'], ['neck', 'body', 'tail_base'])


# CREATE video/pose overlay - loop over videos
for vid in tqdm(videos):
    pose = [p for p in poses if vid.split('.') in p]
    if len(pose) != 1: raise ValueError('ops')
    else: pose = pose[0]

    posedf = pd.read_hdf(os.path.join(dr, pose))

    cap = cv2.VideoCapure(os.path.join(dr, pose))

    framen = 0  # loop over frames
    while True:
        frame, ret = cap.read()
        if not ret: break

        frame_pose = posedf.iloc[framen]

        # plot single bparts
        for bp in frame_pose:
            if bp in bp_groups.head:
                color = colors.head
            elif bp in bp_groups.body:
                color = colors.body
            else:
                continue


            cv2.Circle(frame, (bp.x, bp.y), 5, color, -1)

        # draw mouse polygon
        points = np.array([[frame_pose.rear.x, frame_pose.rear.y], [frame_pose.snout.x, frame_pose.snout.y],
                           [frame_pose.lear.x, frame_pose.lear.y], [frame_pose.tail_base.x, frame_pose.tail_base.y]])
        cv2.fillPoly(frame, points, color=[.2, .2, .2])

        cv2.imshow('frame', frame)
        cv2.WaitKey(1000)






















