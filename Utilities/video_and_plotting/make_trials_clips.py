import sys
sys.path.append('./') 

import os 
import cv2
from tqdm import tqdm

from Utilities.file_io.files_load_save import load_yaml
from Utilities.video_and_plotting.video_editing import Editor
from database.NewTablesDefinitions import *


def create_trials_clips(prestim=10, poststim=20, clean_vids=True):
    def write_clip(video, savename, stim_frame, stim_duration, prestim, poststim, clean_vids):
        # parameters to draw on frame
        border_size = 20
        color_on = [128, 128, 128]
        color_off = [0, 0, 0]
        curr_color = color_off

        # Get video params and open opencv writer
        editor = Editor()
        cap = cv2.VideoCapture(video)
        nframes, width, height, fps = editor.get_video_params(cap)

        writer = editor.open_cvwriter(savename, w=width+border_size*2,
                                      h=height+border_size*2, framerate=fps)

        # Get start and stop frames
        start = stim_frame - prestim*fps
        stop = sitm_frame + poststim*fps
        clip_number_of_frames = stop-start

        # Get stimulus window
        window = (prestim*fps, prestim*fps + stim_duration*fps)

        # Write clip
        for frame_counter in tqdm(range(clip_number_of_frames)):
            ret, frame = cap.read()
            if not ret:
                if abs(frame_counter + start - nframes)<2:  # we are at the end of the clip
                    break
                else:
                    raise ValueError('Something went wrong when opening next frame: {} of {}'.
                                        format(frame_counter, nframes))

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
            videowriter.write(gray)
        videowriter.release()


    # Get path to folder and list of previously saved videos
    paths = load_yaml('./paths.yml')
    save_fld = os.path.join(paths['raw_data_folder'], paths['trial_clips'])
    saved_clips = [f for f in os.listdir(save_fld)]

    # Start looping over Recordings()
    recs = Recordings()
    behav_stims = BehaviourStimuli()
    mantis_simts = MantisStimuli()
    videofiles = VideoFiles()

    for rec in recs.fetc(as_dict=True):
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

                write_clip(videoname, os.path.join(save_fld, clip_name),
                            stim['stim_start'], stim['stim_duration'], 
                            prestim, poststim, clean_vids)


            else:
                raise NotImplementedError

def create_trials_clips(save_fld, rawvideo_fld, rawmetadata_fld, BehavRec=None, folder=None, prestim=10, poststim=20, clean_vids=True):
    """
    This function creates small mp4 videos for each trial and saves them.
    It can work on a single BehaviouralRecording or on a whole folder

    :param BehavRec recording to work on, if None work on a whole folder
    :param folder  path to a folder containing videos to be processed, if None self.raw_video_folder
    """
    if BehavRec:
        raise ValueError('Feature not implemented yet: get trial clips for BehaviourRecording')
    else:
        if folder is None:
            video_fld = rawvideo_fld
            metadata_fld = rawmetadata_fld
        else:
            raise ValueError('Feature not implemented yet: get trial clips for custom folder')

    # parameters to draw on frame
    border_size = 20
    color_on = [128, 128, 128]
    color_off = [0, 0, 0]
    curr_color = color_off

    # LOOP OVER EACH VIDEO FILE IN FOLDER
    metadata_files = os.listdir(metadata_fld)
    for v in os.listdir(video_fld):
        print('\n\n\nProcessing: ', v)
        if os.path.getsize(os.path.join(video_fld, v)) == 0: continue  # skip if video file is empty

        if 'tdms' in v:  # TODO implemente tdms --> avi conversion
            raise ValueError('Feature not implemented yet: get trial clips from .tdms video')
        elif 'txt' in v:
            continue
        else:
            if not 'avi' in v and not 'mp4' in v:
                raise ValueError('Unrecognised video format for : \n', v)
            name = os.path.splitext(v)[0]

            # Check if already processed
            processed = [f for f in os.listdir(save_fld) if name in f]
            if processed: continue

            # Load metadata
            tdms_file = [f for f in metadata_files if name == f.split('.')[0]]
            if len(tdms_file)>1: raise ValueError('Could not disambiguate video --> tdms relationship')
            elif not tdms_file:     # Try a couple of things to rescue this error
                tdms_file = [f for f in metadata_files if name.upper() == f.split('.')[0]]
                if not tdms_file:
                    tdms_file = [f for f in metadata_files if name.lower() == f.split('.')[0]]
                if not tdms_file: # give up
                    raise ValueError('Didnt find a tdms file')
            else:
                # Stimuli frames
                stimuli = load_stimuli_from_tdms(os.path.join(metadata_fld, tdms_file[0]))

                # Open opencv cap reader and extract video metrics
                cap = cv2.VideoCapture(os.path.join(video_fld, v))
                if not cap.isOpened():
                    print('Could not process this one')
                    raise ValueError('Could not load video file')

                fps = cap.get(cv2.CAP_PROP_FPS)
                window = (int(prestim*fps), int(poststim*fps))
                clip_number_of_frames = int(window[1]+window[0])

                # Loop over stims
                for stim_type, stims in stimuli.items():
                    if stim_type == 'audio':
                        stim_end = window[0] + 9 * fps
                    else:
                        stim_end = window[0] + 5 * fps

                    for stim in stims:
                        width, height = int(cap.get(3)), int(cap.get(4))

                        frame_n = stim-window[0]
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n-1)

                        video_path = os.path.join(save_fld, name + '_{}-{}'.format(stim_type, stim) + '.mp4')
                        print('\n\nSaving Clip in: ', video_path)
                        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                        if not clean_vids:
                            videowriter = cv2.VideoWriter(video_path, fourcc, fps, (width + (border_size * 2),
                                                                                height + (border_size * 2)), False)
                        else:
                                videowriter = cv2.VideoWriter(video_path, fourcc, fps, (width , height ), False)

                        for frame_counter in tqdm(range(clip_number_of_frames)):
                            ret, frame = cap.read()
                            if not ret:
                                tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                if frame_counter + frame_n-1 == tot_frames or frame_counter == tot_frames: break
                                else:
                                    raise ValueError('Something went wrong when opening next frame: {} of {}'.
                                                        format(frame_counter, tot_frames))

                            # Prep to display stim on
                            if frame_counter < window[0]:
                                sign = ''
                                curr_color = color_off
                            else:
                                sign = '+'
                                if frame_counter > stim_end: curr_color = color_off
                                else:
                                    if frame_counter % 15 == 0:
                                        if curr_color == color_off: curr_color = color_on
                                        else: curr_color = color_off

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
                            videowriter.write(gray)
                            frame_counter += 1
                        videowriter.release()

                        # Check if file was properly created
                        if not os.path.getsize(video_path) > 1000:  # ! arbitrary value
                            raise ValueError('The file was not created properly')



if __name__ == "__main__":
    paths = load_yaml('./paths.yml')

    create_trials_clips(paths['clip_for_dlc_training'], 
                        os.path.join(paths['raw_data_folder'], paths['raw_video_folder']),
                        os.path.join(paths['raw_data_folder'], paths['raw_metadata_folder']), 
                        BehavRec=None, folder=None, prestim=15, poststim=15, clean_vids=True)


