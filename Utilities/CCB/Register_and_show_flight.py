# Import packages
from video_funcs import peri_stimulus_video_clip, register_arena, get_background
from termcolor import colored

# ========================================================
#           SET PARAMETERS
# ========================================================

# file path of behaviour video
video_file_path = 'Z:\\branco\\Federico\\raw_behaviour\\maze\\video\\180606_CA2762.avi'

# file path of behaviour video
save_file_path = 'C:\\Users\\Federico\\Desktop'

# file path of fisheye correction -- set to an invalid location such as '' to skip fisheye correction
# A corrective mapping for the branco lab's typical camera is included in the repo!
fisheye_map_location = 'gibb.npy'

# frame of stimulus onset
stim_frame = 3000

# seconds before stimulus to start video
window_pre = 5

# seconds before stimulus to start video
window_post = 10

# frames per second of video
fps = 30

# name of experiment
experiment = 'Barnes wall up'

# name of mouse
mouse_id = 'CA3481'

# stimulus type
stim_type = 'visual'

# x and y offset as set in the behaviour software
x_offset = 120
y_offset = 300

# for flight image analysis: darkness relative to background threshold
# (relative darkness, number of dark pixels)
dark_threshold = [.55,950]






# ========================================================
#           GET BACKGROUND
# ========================================================
print(colored('Fetching background', 'green'))
background_image = get_background(
    video_file_path,start_frame=1, avg_over=1)

# ========================================================
#           REGISTER ARENA
# ========================================================
print(colored('Registering arena', 'green'))
registration = register_arena(
    background_image, fisheye_map_location, x_offset, y_offset)

# ========================================================
#           SAVE CLIPS AND FLIGHT IMAGES
# ========================================================
print(colored('Creating flight clip and image', 'green'))
start_frame = int(stim_frame-(window_pre*fps))
stop_frame = int(stim_frame+(window_post*fps))

videoname = '{}_{}_{}-{}\''.format(experiment,mouse_id,stim_type, round(stim_frame / fps / 60))

peri_stimulus_video_clip(video_file_path, videoname, save_file_path, start_frame, stop_frame, stim_frame,
                         registration, x_offset, y_offset, dark_threshold,
                         save_clip = True, display_clip = True, counter = True, make_flight_image = True)
