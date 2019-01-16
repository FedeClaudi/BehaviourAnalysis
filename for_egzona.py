import sys
sys.path.append('./')  

import warnings as warn
try: import cv2
except: pass
# from moviepy.editor import VideoFileClip, concatenate_videoclips
# from moviepy.video.fx import crop
import os
from tempfile import mkdtemp
from tqdm import tqdm
from collections import namedtuple
from nptdms import TdmsFile
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import shutil
import matplotlib.pyplot as plt
import time
import pandas as pd

from Utilities.file_io.files_load_save import load_yaml, load_tdms_from_winstore
from Processing.plot.plotting_utils import get_bps_as_points_dict


def plot_for_egzona(txt, pfile, ax):
    colors = dict(back = ([.8, .4, .4], .2),
                left_FL = ([.4, .8, .4], .8),
                left_HL = ([.2, .4, .2], .3), 
                neck = ([.6, .3, .3], .2), 
                right_FL = ([.4, .4, .8], .8),
                right_HL = ([.2, .2, .4], .3),
                snout = ([.8, .5, .4], .5), 
                tail1 = ([.4, .6, .4], .3), 
                tail2 = ([.4, .6, .5], .3), 
                tail3 = ([.4, .6, .6], .3),
                tail4 = ([.4, .7, .7], .3), 
                tail5 = ([.4, .8, .8], .3))
    scorer = 'DeepCut_resnet50_forceplateJan14shuffle1_930000'
    pose = pd.read_hdf(pfile)
    pose = pose[scorer]

    # f, ax = plt.subplots()
    frames = []
    with open(txt, 'r') as f:
        frames = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    frames = [x.strip() for x in frames]
    frames = [int(f) for f in frames if f ]
    
    frame_groups = [np.linspace(f, f+30*1, 30*1+1) for f in frames]
    for frame_group in frame_groups:
        temp_pose = pose.iloc[frame_group]
        for bp in colors.keys():
            if bp == 'neck': continue
            elif 'tail' in bp.lower(): continue
            # ax.scatter(temp_pose[bp]['x'].values, -temp_pose[bp]['y'].values, s=35, c=colors[bp][0], alpha=colors[bp][1], label=bp)
            ax.plot(temp_pose[bp]['x'].values, -temp_pose[bp]['y'].values, color=colors[bp][0], alpha=colors[bp][1], label=bp)
            ax.set(facecolor=[.2, .2, .2])
            ax.legend()
        break













if __name__ == "__main__":
    fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\DLC_nets\\Egzona"
    
    
    text_files = [f for f in os.listdir(fld) if '.txt' in f] 
    for i, txt in enumerate(text_files):
        if i == 0:
            f, axarr = plt.subplots(len(text_files), 1)

        print(txt)
        posef = txt[:-12]
        
        pose = os.path.join(fld, posef+'.h5')
        plot_for_egzona(os.path.join(fld, txt), pose, axarr[i])

    plt.show()
