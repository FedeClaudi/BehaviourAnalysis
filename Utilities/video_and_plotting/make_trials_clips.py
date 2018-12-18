import sys
sys.path.append('./') 

import os 
import cv2
from tqdm import tqdm
import pandas as pd

from Utilities.file_io.files_load_save import load_yaml
from Utilities.video_and_plotting.video_editing import Editor
from database.NewTablesDefinitions import *
from Processing.plot.plotting_utils import *
from Processing.plot.video_plotting_toolbox import *


def create_trials_clips(prestim=10, poststim=20, clean_vids=True, plt_pose=False):
    def write_clip(video, savename, stim_frame, stim_duration, prestim, poststim, clean_vids, posedata):
        # parameters to draw on frame
        border_size = 20
        color_on = [128, 128, 128]
        color_off = [0, 0, 0]
        curr_color = color_off

        # Get video params and open opencv writer
        editor = Editor()
        cap = cv2.VideoCapture(video)
        if not cap.isOpened(): raise FileNotFoundError(video)
        nframes, width, height, fps = editor.get_video_params(cap)

        writer = editor.open_cvwriter(savename, w=width+border_size*2,
                                      h=height+border_size*2, framerate=fps)

        # Get start and stop frames
        start = stim_frame - prestim*fps
        stop = stim_frame + poststim*fps
        clip_number_of_frames = int(stop-start)

        # Get stimulus window
        window = (prestim*fps, prestim*fps + stim_duration*fps)

        # Set cap to correct frame number
        real_start_frame = stim_frame - int(prestim*fps)
        cap.set(1, real_start_frame)


        # Write clip
        for frame_counter in tqdm(range(clip_number_of_frames)):
            ret, frame = cap.read()
            if not ret:
                if abs(frame_counter + start - nframes)<2:  # we are at the end of the clip
                    break
                else:
                    raise ValueError('Something went wrong when opening next frame: {} of {}'.
                                        format(frame_counter, nframes))

            # Overylay bodypart position of frame
            if posedata is not None and not clean_vids:
                real_frame_number = real_start_frame + frame_counter
                frame_pose = posedata.iloc[real_frame_number]
                points_dict = get_bps_as_points_dict(frame_pose)
                frame = cv2_plot_mouse_bps(frame, points_dict, s=2)

            # Prep to display stim on
            if frame_counter < window[0] or frame_counter > window[1]:
                sign = ''
                curr_color = color_off
            else:
                sign = '+'
                curr_color = color_on
                
            # Make frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not clean_vids:
                gray = cv2.copyMakeBorder(gray, border_size, border_size, border_size, border_size,
                                            cv2.BORDER_CONSTANT, value=curr_color)

                frame_time = (frame_counter - window[0]) / fps
                frame_time = str(round(.2 * round(frame_time / .2), 1)) + '0' * (abs(frame_time) < 10)
                cv2.putText(gray, sign + str(frame_time) + 's', (width - 110, height + 10), 0, 1,
                            (180, 180, 180), thickness=2)

            # Save to file
            writer.write(gray)
        writer.release()

    # Get path to folder and list of previously saved videos
    paths = load_yaml('./paths.yml')
    save_fld = os.path.join(paths['raw_data_folder'], paths['trials_clips'])
    saved_clips = [f for f in os.listdir(save_fld)]

    # Start looping over Recordings()
    recs = Recordings()
    behav_stims = BehaviourStimuli()
    mantis_simts = MantisStimuli()
    videofiles = VideoFiles()

    for rec in recs.fetch(as_dict=True):
        # Get the stim table entry and clip ame

        if rec['software'] == 'behaviour':
            stims = [s for s in behav_stims if s['recording_uid']==rec['recording_uid']]
        else:
            stims = [s for s in mantis_simts if s['recording_uid'] == rec['recording_uid']]
        
        for stim in stims:
            clip_name = stim['stimulus_uid']+'.mp4'
            if clip_name in saved_clips: continue  # avoid doing again an old clip
                
            # Get frame time and video name for behaviour
            if rec['software'] == 'behaviour':
                videoname = stim['video']

                
                if plt_pose:
                    # Get pose data
                    videoentry = [v for v in videofiles if v['video_filepath']==videoname or v['converted_filepath'] == videoname][0]
                
                    posefile = videoentry['pose_filepath']
                    try:
                        posedata = pd.read_hdf(posefile)
                    except:
                        raise FileNotFoundError(posefile)
                else:
                    posedata = None

                # Write clip
                write_clip(videoname, os.path.join(save_fld, clip_name),
                            stim['stim_start'], stim['stim_duration'], 
                            prestim, poststim, clean_vids, posedata)


            else:
                raise NotImplementedError


if __name__ == "__main__":
    paths = load_yaml('./paths.yml')

    create_trials_clips(prestim=10, poststim=20, clean_vids=False)

