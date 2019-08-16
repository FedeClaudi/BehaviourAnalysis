# %%
import sys
sys.path.append("./")
import deeplabcut as dlc
import os

fld = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Scotland"
videos_fld = "/Volumes/Elements/scotland_filming/Day1/B/Clip"

videos = [os.path.join(videos_fld, v) for v in os.listdir(videos_fld) if ".MXF" in v]
print(videos)

# %%
# # ? create proj
dlc.create_new_project("scotland", "steve", 
    videos, working_directory=fld, copy_videos=False)

# %%
# ? run stuff
config_file = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Scotland/scotland-steve-2019-08-12/config.yaml"

# vids_to_add = ["/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/ants/vids/Decision Making Videos/100L60R/Ant6_Crop.avi",
#                 "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/ants/vids/Decision Making Videos/120L60R/Ant5_Run1_Crop30fps.avi",
#                 "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/ants/vids/Decision Making Videos/140L60R/Ant4_LeftCrop_30fps.avi",
#                 ]

# dlc.add_new_videos(config_file, vids_to_add, copy_videos=True)

dlc.extract_frames(config_file, mode="manual")
# dlc.label_frames(config_file)

# dlc.create_training_dataset(config_file)
# dlc.train_network(config_file)

# dlc.analyze_videos(config_file, videos=videos, videotype="mp4")
# dlc.create_labeled_video(config_file, videos, draw_skeleton=True)

