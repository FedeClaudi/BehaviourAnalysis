# %%
# Imports
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2

from fcutils.file_io.io import load_yaml
from fcutils.file_io.utils import get_file_name
from fcutils.video.utils import get_cap_from_file, get_cap_selected_frame, get_video_params, open_cvwriter

from behaviour.tdms.utils import get_analog_inputs_clean_dataframe
from behaviour.utilities.signals import get_times_signal_high_and_low, convert_from_sample_to_frame
import multiprocessing as mp

# Vars
fps = 40
n_frames_pre = 5 * fps
n_frames_pos = 15 * fps

output_fld = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\shortctu\\trials'
notes_path = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\shortctu\\notes.yml'
notes = load_yaml(notes_path)

# Text stuff
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,950)
fontScale              = 1
lineType               = 2

# Define variables
videos_fld = 'Z:\\swc\\branco\\Federico\\raw_behaviour\\maze\\video'
ai_fld = 'Z:\\swc\\branco\\Federico\\raw_behaviour\\maze\\analoginputdata'

sessions = ['200227_CA8755_1', '200227_CA8754_1', '200227_CA8753_1', '200227_CA8752_1',
            '200225_CA8751_1', '200225_CA848_1', '200225_CA8483_1', '200225_CA834_1',
            '200225_CA832_1', '200210_CA8491_1', '200210_CA8482_1', '200210_CA8481_1',
            '200210_CA8472_1', '200210_CA8471_1', '200210_CA8283_1']

videos = [os.path.join(videos_fld, s+'Overview.mp4') for s in sessions]
ais = [os.path.join(ai_fld, s+'.tdms') for s in sessions]


# %%
def process_session(args):
    video, aifile = args

    # session name
    name = get_file_name(video).split("Overview")[0]
    if notes[name]['overall'] != 'keep':
        return
    # Get save path and video writer
    savepath = os.path.join(output_fld, get_file_name(video)+'_trials.mp4')
    # if os.path.isfile(savepath):
    #     return
    # else:
    print(f'Processing {get_file_name(video)}\n')

    # Open video
    videocap = get_cap_from_file(video)
    nframes, width, height, orig_fps = get_video_params(videocap)
    writer = open_cvwriter(savepath, w=width, h=height, framerate=fps, iscolor=True)
    
    # Get analog inputs
    ai = get_analog_inputs_clean_dataframe(aifile, save_df=False)

    # Get stim onset frames
    onsets, offsets = get_times_signal_high_and_low(ai.AudioFromSpeaker_AI, th=2.25, min_time_between_highs=10000)
    onsets_frame = convert_from_sample_to_frame(onsets, 250000, 400)

    # Loop over stimuli
    for stimn, stim in enumerate(onsets_frame):
        print(f"    {get_file_name(video)} -  stim {stimn} of {len(onsets_frame)}")
        for framen in np.arange(stim-n_frames_pre, stim+n_frames_pos):
            frame = get_cap_selected_frame(videocap, int(framen))

            if frame is None: break

            if framen >= stim and framen <= stim+9*fps:
                cv2.circle(frame, (700, 75), 50, (0, 255, 0), -1)

            cv2.putText(frame, f'Stim {stimn} of {len(onsets_frame)-1}', 
                (50, 50), 
                font, 
                fontScale,
                (30, 220, 30),
                lineType)

            writer.write(frame.astype(np.uint8))
    writer.release()

# %%
if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count()-2)
    # results = [pool.apply(process_session, args=(video, aifile)) for video, aifile in zip(videos, ais)]
    pool.map(process_session, [(vid, ai) for vid, ai in zip(videos, ais)])
    pool.close()