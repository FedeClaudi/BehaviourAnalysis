# %%
import sys
sys.path.append("./")
import deeplabcut as dlc
import os


fld = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/ants/dlc"
videos_fld = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/ants/dlc_vids"

edited_videos = [v for v in os.listdir(videos_fld) if "_edited" in v]
videos = [os.path.join(videos_fld, v) for v in os.listdir(videos_fld) if v not in edited_videos]

# ? edit vidoes:
# from Utilities.video_and_plotting.video_editing import Editor
# editor = Editor()
# for vid in videos:
#     f, name = os.path.split(vid)
#     name = name.split(".")[0]+"_edited.mp4"
#     editor.compress_clip(vid, .5, save_path=os.path.join(f, name))
# %%

# # ? video projs
# videos = [os.path.join(videos_fld, v) for v in os.listdir(videos_fld) if "edited"in v]

# # ? create proj
# dlc.create_new_project("ants", "federico", 
#     videos, working_directory=fld, copy_videos=True)

# ? run stuff
config_file = '/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/ants/dlc/ants-federico-2019-06-24/config.yaml'

# dlc.add_new_videos(config_file, ["/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/ants/dlc_vids/single_zoomed_2_edited.mp4"], copy_videos=True)

# dlc.extract_frames(config_file, "manual",)
# dlc.label_frames(config_file)

dlc.create_training_dataset(config_file)
# dlc.train_network(config_file)

# dlc.analyze_videos(config_file, videos=videos, videotype="mp4")
# dlc.create_labeled_video(config_file, videos, draw_skeleton=True)

