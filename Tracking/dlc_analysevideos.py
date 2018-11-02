"""
Video analysis using a trained network, based on code by
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu
"""

import os.path
import sys
from  Utils.loadsave_funcs import load_paths
paths = load_paths()
dlc_folder = paths['DLC folder']

# add parent directory: (where nnet & config are!)
sys.path.append(os.path.join(dlc_folder, "pose-tensorflow"))
sys.path.append(os.path.join(dlc_folder, "Generating_a_Training_Set"))

from Tracking.dlc_analysis_config import cropping, x1, x2, y1, y2, videotype

# Deep-cut dependencies
from nnet import predict
from dataset.pose_dataset import data_to_input


import pickle
import imageio
imageio.plugins.ffmpeg.download()
from skimage.util import img_as_ubyte
from moviepy.editor import VideoFileClip
import skimage
import skimage.color
import time
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def getpose(sess, inputs, image, cfg, outputs, outall=False):
    ''' Adapted from DeeperCut, see pose-tensorflow folder'''
    image_batch = data_to_input(skimage.color.gray2rgb(image))
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref = predict.extract_cnn_output(outputs_np, cfg)
    pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
    if outall:
        return scmap, locref, pose
    else:
        return pose


def analyse(tf_setting, videofolder:str, clips_l:list):
    """  analyse the videos in videofolder that are also listed in clips_l"""
    # Load TENSORFLOW settings
    cfg = tf_setting['cfg']
    scorer = tf_setting['scorer']
    sess = tf_setting['sess']
    inputs = tf_setting['inputs']
    outputs = tf_setting['outputs']

    pdindex = pd.MultiIndex.from_product(
        [[scorer], cfg['all_joints_names'], ['x', 'y', 'likelihood']],
        names=['scorer', 'bodyparts', 'coords'])
    frame_buffer = 10

    os.chdir(videofolder)
    videos = np.sort([fn for fn in os.listdir(os.curdir) if (videotype in fn)])
    clips_l = [item for sublist in clips_l for item in sublist]
    for video in videos:
        try:
            if video.split('.')[0] not in clips_l:
                continue

            dataname = video.split('.')[0] + scorer + '.h5'
            try:
                # Attempt to load data...
                pd.read_hdf(dataname)
                print("            ... video already analyzed!", dataname)
            except FileNotFoundError:
                print("                 ... loading ", video)
                clip = VideoFileClip(video)
                ny, nx = clip.size  # dimensions of frame (height, width)
                fps = clip.fps
                nframes_approx = int(np.ceil(clip.duration * clip.fps) + frame_buffer)

                if cropping:
                    clip = clip.crop(
                        y1=y1, y2=y2, x1=x1, x2=x2)  # one might want to adjust

                start = time.time()
                PredicteData = np.zeros((nframes_approx, 3 * len(cfg['all_joints_names'])))

                temp_image = img_as_ubyte(clip.get_frame(0))
                scmap, locref, pose = getpose(sess, inputs, temp_image, cfg, outputs, outall=True)
                PredictedScmap = np.zeros((nframes_approx, scmap.shape[0], scmap.shape[1], len(cfg['all_joints_names'])))

                for index in tqdm(range(nframes_approx)):
                    image = img_as_ubyte(clip.reader.read_frame())

                    if index == int(nframes_approx - frame_buffer * 2):
                        last_image = image
                    elif index > int(nframes_approx - frame_buffer * 2):
                        if (image == last_image).all():
                            nframes = index
                            print("Detected frames: ", nframes)
                            break
                        else:
                            last_image = image
                    try:
                        pose = getpose(sess, inputs,image, cfg, outputs, outall=True)
                        PredicteData[index, :] = pose.flatten()
                    except:
                        scmap, locref, pose = getpose(sess, inputs, image, cfg, outputs, outall=True)
                        PredicteData[index, :] = pose.flatten()
                        PredictedScmap[index, :, :, :] = scmap

                stop = time.time()

                dictionary = {
                    "start": start,
                    "stop": stop,
                    "run_duration": stop - start,
                    "Scorer": scorer,
                    "config file": cfg,
                    "fps": fps,
                    "frame_dimensions": (ny, nx),
                    "nframes": nframes
                }
                metadata = {'data': dictionary}

                print("Saving results...")
                DataMachine = pd.DataFrame(PredicteData[:nframes, :], columns=pdindex,
                                           index=range(nframes))  # slice pose data to have same # as # of frames.
                DataMachine.to_hdf(dataname, 'df_with_missing', format='table', mode='w')

                with open(dataname.split('.')[0] + 'includingmetadata.pickle',
                          'wb') as f:
                    pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)
        except:
            from warnings import warn
            warn('Could not do DLC tracking on video {}'.format(video))


