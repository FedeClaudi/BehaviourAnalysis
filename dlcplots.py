from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import os

make_videos = False
make_plots = True

dr = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\DAQ\\upstairs_rig\\video_clips\\videos_for_FC'
videoname = "Barnes US wall up (2)_CA3380_audio-3 (27')"
posename = "Barnes US wall up (2)_CA3380_audio-3 (27')DeepCut_resnet50_BarnesNov9shuffle1_150000_labeled"

videos = [f for f in os.listdir(dr) if'avi' in f and 'labeled' not in f]
posesfiles = [f for f in os.listdir(dr) if 'h5' in f and 'metadata' not in f]
# videos = [os.path.join(dr, videoname)]
#poses = [os.path.join(dr, posename)]

cols = namedtuple('colors', 'head body')
colors = cols([255, 128, 50], [50, 128, 255])
plotcolors = cols([.8, .5, .2], [.2, .7, .2])
bp_groups = cols(['snout', 'rear', 'lear'], ['neck', 'body', 'tail_base'])


# Create frames plots
if make_plots:
    for video in videos:
        posefile = [p for p in posesfiles if video.split('.')[0] in p][0]

        print('plotting: ', video)

        f, ax = plt.subplots(frameon=False, figsize=(10, 8))
        ax.set(facecolor=[.2, .2, .2], ylim=[650, 0], xticks=[], yticks=[])
        posedf = pd.read_hdf(os.path.join(dr, posefile))
        poses = posedf['DeepCut_resnet50_BarnesNov9shuffle1_150000']

        poses_range = range(280, 350)
        positions = np.linspace(0, 1200, len(range(280, 350)))
        for i, framen in enumerate(poses_range):
            if framen == 300:
                size = 120
                lwidth = 10
                alpha = 1
                ax.axvline(positions[i], color='r', linewidth=3, alpha=.75)
            elif framen % 5 == 0:
                size = 120
                lwidth = 10
                alpha = 1
                ax.axvline(positions[i], color='w', linewidth=1, alpha=.75)
            else:
                size = 25
                lwidth = 3
                alpha = .07

            frame_pose = poses.iloc[framen]

            plotted = []

            spacer = frame_pose.body.x - positions[i]
            pointsdict = {}
            for bpname in frame_pose.keys():
                bpname = bpname[0]
                if bpname in plotted: continue
                bp = frame_pose[bpname]
                if bpname in bp_groups.head:
                    color = plotcolors.head
                elif bpname in bp_groups.body:
                    color = plotcolors.body
                else:
                    continue

                ax.scatter(bp.x-spacer, bp.y, color=color, s=40, alpha=alpha)
                pointsdict[bpname] = [bp.x, bp.y]

            linesparts = [('lear', 'snout'), ('snout', 'rear'), ('rear', 'neck'), ('lear', 'neck'),
                          ('body', 'neck'), ('body', 'tail_base')]
            for l1, l2 in linesparts:
                if l1 in bp_groups.head:
                    color = plotcolors.head
                else:
                    color = plotcolors.body
                ax.plot([pointsdict[l1][0]-spacer, pointsdict[l2][0]-spacer],
                        [pointsdict[l1][1], pointsdict[l2][1]], color=color, linewidth=3, alpha=alpha)

        f.savefig(os.path.join(dr, video.split('.')[0])+'.eps', format='eps', dpi=1000)

# CREATE video/pose overlay - loop over videos
if make_videos:
    for vid in tqdm(videos):
        # pose = [p for p in poses if vid.split('.') in p]
        pose = poses
        if len(pose) != 1: raise ValueError('ops')
        else: pose = pose[0]

        posedf = pd.read_hdf(os.path.join(dr, pose, ))

        cap = cv2.VideoCapture(vid)

        savepath1 = vid.split('.')[0] + '_edited_points' + '.mp4'
        savepath2 = vid.split('.')[0] + '_edited_liness' + '.mp4'
        savepath3 = vid.split('.')[0] + '_edited_polys' + '.mp4'

        fourcc = cv2.VideoWriter_fourcc(*'X264')
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        wpoints = cv2.VideoWriter(savepath1, fourcc, 30, (h, w), False)
        wlines =  cv2.VideoWriter(savepath2, fourcc, 30, (h, w), False)
        wpolys =  cv2.VideoWriter(savepath3, fourcc, 30, (h, w), False)

        framen = 0  # loop over frames
        while True:
            print('frame n: ', framen)
            ret, frame = cap.read()
            blkframe =np.uint8( np.zeros(frame.shape))
            frame2 = frame.copy()
            if not ret: break
            frame_pose = posedf.iloc[framen]

            # plot single bparts
            frame_pose = frame_pose['DeepCut_resnet50_BarnesNov9shuffle1_150000']

            plotted  =[]
            pointsdict = {}
            for bpname in frame_pose.keys():
                bpname = bpname[0]
                if bpname in plotted: continue
                bp = frame_pose[bpname]
                if bpname in bp_groups.head:
                    color = colors.head
                elif bpname in bp_groups.body:
                    color = colors.body
                else:
                    continue
                cv2.circle(frame, (np.int32(bp.x), np.int32(bp.y)), 5, color, -1)
                pointsdict[bpname] = np.int32([bp.x, bp.y])

            # draw mouse polygon
            cv2.fillPoly(blkframe,  [np.int32([pointsdict[k] for k in bp_groups.body])], color=colors.body)
            cv2.fillPoly(blkframe, [np.int32([pointsdict[k] for k in bp_groups.head])], color=colors.head)

            cv2.polylines(frame2, [np.int32([pointsdict[k] for k in ['lear', 'body', 'rear']])], color=[0, 255, 0], isClosed=True)
            cv2.polylines(frame2, [np.int32([pointsdict[k] for k in ['body', 'tail_base']])], color=[0, 255, 0], isClosed=True)

            cv2.polylines(frame2, [np.int32([pointsdict[k] for k in bp_groups.head])], color=[0, 255, 0], isClosed=True)

            cv2.imshow('blkframe', frame2)
            cv2.waitKey(1)

            wpoints.write(frame)
            wpolys.write(blkframe)
            wlines.write(frame2)

            framen += 1
            if framen > 350: break
        wpoints.release()
        wpolys.release()
        wlines.release()




















