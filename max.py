import numpy as np
import os
import cv2


def get_video_params(cap):
    if isinstance(cap, str):
        cap = cv2.VideoCapture(cap)
        
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    return nframes, width, height, fps 

def open_cvwriter(filepath, w=None, h=None, framerate=None, format='.mp4', iscolor=False):
    try:
        if 'avi' in format:
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')    # (*'MP4V')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videowriter = cv2.VideoWriter(filepath, fourcc, framerate, (w, h), iscolor)
    except:
        raise ValueError('Could not create videowriter')
    else:
        return videowriter

def trim_clip(videopath, savepath, 
                start_frame=None, stop_frame=None):
    """trim_clip [take a videopath, open it and save a trimmed version between start and stop. Either 
    looking at a proportion of video (e.g. second half) or at start and stop frames]
    
    Arguments:
        videopath {[str]} -- [video to process]
        savepath {[str]} -- [where to save]
    
    Keyword Arguments:
        start_frame {[type]} -- [video frame to stat at ] (default: {None})
        end_frame {[type]} -- [videoframe to stop at ] (default: {None})
    """

    # Open reader and writer
    cap = cv2.VideoCapture(videopath)
    nframes, width, height, fps  = get_video_params(cap)
    writer = open_cvwriter(savepath, w=width, h=height, framerate=int(fps), format='.mp4', iscolor=False)

    # Loop over frames and save the ones that matter
    print('Processing: ', videopath)
    cur_frame = 0
    cap.set(1,start_frame)
    while True:
        cur_frame += 1
        if cur_frame % 100 == 0: print('Current frame: ', cur_frame)
        if cur_frame <= start_frame: continue
        elif cur_frame >= stop_frame: break
        else:
            ret, frame = cap.read()
            if not ret: break
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                writer.write(frame)
    writer.release()