import sys
sys.path.append('./') 

import os 
import cv2
from collections import namedtuple
import numpy as np

from Utilities.video_editing import Editor


def run():
    fld = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\3dcam_test'


    # cropping params
    crop = namedtuple('coords', 'x0 x1 y0 y1')
    main = crop(320, 1125, 250, 550)
    side = crop(1445, 200, 250, 550)
    top = crop(675, 800, 75, 200)

    edit = Editor

    videos = sorted(os.listdir(fld))
    for v in videos:
        if '.mp4' in v: continue
        orig = os.path.join(fld, v)
        cap = cv2.VideoCapture(orig)

        main_name = os.path.join(os.path.join(fld, '{}.mp4'.format(v.split('.')[0]+'_main')))
        main_writer = edit.open_cvwriter(filepath=main_name, w=main.x1, h=main.y1, framerate=30)
        side_name = os.path.join(os.path.join(fld, '{}.mp4'.format(v.split('.')[0]+'_side')))
        side_writer = edit.open_cvwriter(filepath=side_name, h=side.x1, w=side.y1, framerate=30)
        top_name = os.path.join(os.path.join(fld, '{}.mp4'.format(v.split('.')[0]+'_top')))
        top_writer = edit.open_cvwriter(filepath=top_name, w=top.x1, h=top.y1, framerate=30)

        writers = [main_writer, side_writer, top_writer]

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            main_frame = frame[main.y0:main.y0+main.y1, main.x0:main.x0+main.x1]
            side_frame = frame[side.y0:side.y0+side.y1, side.x0:side.x0+side.x1]

            print(side_frame.shape)
            side_frame = np.rot90(side_frame, 1)
            print(side_frame.shape)

            top_frame = frame[top.y0:top.y0+top.y1, top.x0:top.x0+top.x1]

            cv2.imshow('main', main_frame)
            cv2.imshow('side', side_frame)
            cv2.imshow('top', top_frame)

            main_writer.write(main_frame)
            side_writer.write(side_frame)
            top_writer.write(top_frame)

            cv2.waitKey(1)
        
        for wr in writers:
            wr.release()

if __name__ == "__main__":
    run()