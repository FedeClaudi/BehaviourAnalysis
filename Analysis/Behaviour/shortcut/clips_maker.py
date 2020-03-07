# %%
# Importd
import os
import sys

from tqdm import tqdm

from fcutils.video.utils import get_cap_from_file, get_cap_selected_frame, get_video_params, open_cvwriter

from behaviour.tdms.utils import get_analog_inputs_clean_dataframe



# %%
# ------------------------------ Fetch filepaths ----------------------------- #
if sys.platform != 'darwin':
    videos_fld = 'Z:\\swc\\branco\\Federico\\raw_behaviour\\maze\\video'
    ai_fld = 'Z:\\swc\\branco\\Federico\\raw_behaviour\\maze\\analoginputdata'
else:
    videos_fld = '/Volumes/swc/branco/Federico/raw_behaviour/maze/video'
    ai_fld = '/Volumes/swc/branco/Federico/raw_behaviour/maze/analoginputdata'

sessions = ['200227_CA8755_1', '200227_CA8754_1', '200227_CA8753_1', '200227_CA8752_1',
            '200225_CA8751_1', '200225_CA848_1', '200225_CA8483_1', '2/00225_CA834_1',
            '200225_CA832_1', '200210_CA8491_1', '200210_CA8482_1', '200210_CA8481_1',
            '200210_CA8472_1', '200210_CA8471_1', '200210_CA8283_1']

videos = [os.path.join(videos_fld, s+'Overview.mp4') for s in sessions]
ais = [os.path.join(ai_fld, s+'.tdms') for s in sessions]

# %%
# -------------------------------- Fetch data -------------------------------- #
session_n = 0

ai = get_analog_inputs_clean_dataframe(ais[session_n], save_df=True)
videocap = get_cap_from_file(videos[session_n])
nframes, width, height, fps = get_video_params(videocap)

# %%
# ---------------------------- Find stimuli times ---------------------------- #

			# Get stim times from audio channel data
			if  'AudioFromSpeaker_AI' in groups:
				audio_channel_data = tdms_df.channel_data('AudioFromSpeaker_AI', '0')
				th = 1
			else:
				# First recordings with mantis had different params
				audio_channel_data = tdms_df.channel_data('AudioIRLED_AI', '0')
				th = 1.5
			
			# Find when the stimuli start in the AI data
			stim_start_times = find_audio_stimuli(audio_channel_data, th, table.sampling_rate)