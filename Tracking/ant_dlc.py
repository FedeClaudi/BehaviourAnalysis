# %%
import sys
sys.path.append("./")
import deeplabcut as dlc
import os

if sys.platform == "darwin":
    fld = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/ants/ants-dlc"
    videos_fld = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/ants/vids"
else:
    fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\ants\\ants-dlc"
    videos_fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\ants\\vids"


# %%
# # ? create proj
# videos = os.path.join(videos_fld, v) for v in os.listdir(videos_fld)]
# print(videos)
# dlc.create_new_project("scotland", "steve", videos, working_directory=fld, copy_videos=False) # ? edit this

# %%
# ? config file
config_file =  os.path.join(fld, "config.yaml")
if not os.path.isfile(config_file):
    raise FileExistsError(config_file)


    
# %%
# ? Add vids to proj
# vids_to_add = ["/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/ants/vids/Decision Making Videos/100L60R/Ant6_Crop.avi",
#                 "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/ants/vids/Decision Making Videos/120L60R/Ant5_Run1_Crop30fps.avi",
#                 "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/ants/vids/Decision Making Videos/140L60R/Ant4_LeftCrop_30fps.avi",
#                 ]
# dlc.add_new_videos(config_file, vids_to_add, copy_videos=True)

# %% 
# ? extract frames and label
# dlc.extract_frames(config_file, mode="manual")
# dlc.label_frames(config_file)

# %%
# ? Create dataset and train
# dlc.create_training_dataset(config_file)
dlc.train_network(config_file)

# %%
# ? Analyse
# analysis_fld = os.path.join(videos_fld, "Decision Making Videos\\100L60R")
# videos = [os.path.join(analysis_fld, v) for v in os.listdir(analysis_fld) if ".avi" in v]
# dlc.analyze_videos(config_file, videos=videos, videotype="avi")
# dlc.create_labeled_video(config_file, videos)



#%%
