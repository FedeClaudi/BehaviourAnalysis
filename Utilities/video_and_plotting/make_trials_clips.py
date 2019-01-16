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


def create_trials_clips(prestim=10, poststim=10, clean_vids=True, plt_pose=False):
    def write_clip(video, savename, stim_frame, stim_duration, prestim, poststim, clean_vids, posedata):
        # parameters to draw on frame
        border_size = 0 # 20
        color_on = [100, 255, 100]
        color_off = [20,20,20]
        curr_color = color_off

        # Get video params and open opencv writer
        editor = Editor()
        cap = cv2.VideoCapture(video)
        if not cap.isOpened(): 
            # return
            raise FileNotFoundError(video)
        nframes, width, height, fps = editor.get_video_params(cap)

        writer = editor.open_cvwriter(savename, w=width+border_size*2,
                                      h=height+border_size*2, framerate=fps, iscolor=True)

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
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = frame
            if not clean_vids:
                # gray = cv2.copyMakeBorder(gray, border_size, border_size, border_size, border_size,
                #                             cv2.BORDER_CONSTANT, value=curr_color)
                cv2.circle(gray, (width-200, height-200), 30, curr_color, -1)
                frame_time = (frame_counter - window[0]) / fps
                frame_time = str(round(.2 * round(frame_time / .2), 1)) + '0' * (abs(frame_time) < 10)
                cv2.putText(gray, sign + str(frame_time) + 's', (width - 250, height - 130), 0, 1,
                            (20, 20, 20), thickness=2)

            # frame[:, :, 0] = gray

            # Save to file
            writer.write(gray)
        writer.release()

    # Get path to folder and list of previously saved videos
    paths = load_yaml('./paths.yml')
    # save_fld = os.path.join(paths['raw_data_folder'], paths['trials_clips'])
    save_fld = paths['trials_clips']
    saved_clips = [f for f in os.listdir(save_fld)]

    # Start looping over Recordings()
    recs = Recordings()
    behav_stims = BehaviourStimuli()
    mantis_simts = MantisStimuli()
    videofiles = VideoFiles()
    
    videos_df = pd.DataFrame(videofiles.fetch())

    for recn, rec in enumerate(recs.fetch(as_dict=True)):
        # Get the stim table entry and clip ame
        print('Processing recording {} of {}'.format(recn, len(recs.fetch())))
        if rec['uid']<194: 
            print(' ... skipped')
            continue

        if rec['software'] == 'behaviour':
            stims = [s for s in behav_stims if s['recording_uid']==rec['recording_uid']]
        else:
            stims = [s for s in mantis_simts if s['recording_uid'] == rec['recording_uid']]
        
        for stimn, stim in enumerate(stims):
            print('     stim {} of {}'.format(stimn, len(stims)))
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
                # Get the corrisponding videofile
                entry = videos_df.loc[(videos_df['recording_uid'] == stim['recording_uid']) & (videos_df['camera_name'] == 'overview')]
                videoname = entry['converted_filepath'].values[0]
                fld, name = os.path.split(videoname)

                correct_name =  name.split('__')[0]  # ! only necessary until database entry fixed
                clip_name = name.split('__')[0]+'_{}.mp4'.format(stimn)
                raise ValueError(clip_name)
                
                print('Saving : ', os.path.join(save_fld, clip_name))
                dur = stim['duration']*120  # ! hardcoded duration in fps
                write_clip(os.path.join(fld, correct_name), os.path.join(save_fld, clip_name),
                            stim['overview_frame'], dur, 
                            prestim, poststim, clean_vids, None)

                
                

            


if __name__ == "__main__":
    paths = load_yaml('./paths.yml')

    create_trials_clips(clean_vids=True)

