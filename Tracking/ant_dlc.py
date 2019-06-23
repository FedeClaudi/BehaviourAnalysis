# %%
import sys
sys.path.append("./")
import deeplabcut as dlc
import os

fld = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/ants/dlc"
videos_fld = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/ants/dlc_vids"
videos = [os.path.join(videos_fld, v) for v in os.listdir(videos_fld)]

# ? edit vidoes:
# from Utilities.video_and_plotting.video_editing import Editor
# editor = Editor()
# for vid in videos:
#     f, name = os.path.split(vid)
#     name = name.split(".")[0]+"_edited.mp4"
#     editor.compress_clip(vid, .5, save_path=os.path.join(f, name))

# ? video projs
# videos = [os.path.join(videos_fld, v) for v in os.listdir(videos_fld) if "edited"in v]

# ? create proj
# dlc.create_new_project("ants", "federico", 
#     videos, working_directory=fld, copy_videos=True)

# ? run stuff
config_file = '/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/ants/dlc/ants-federico-2019-06-23/config.yaml'

# dlc.extract_frames(config_file, "manual",)
# dlc.label_frames(config_file)
# dlc.create_training_dataset(config_file)
# dlc.train_network(config_file)

dlc.analyze_videos(config_file, videos=videos, videotype="mp4")
dlc.create_labeled_video(config_file, videos, draw_skeleton=True)

