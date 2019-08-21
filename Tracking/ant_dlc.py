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
# videos = [os.path.join(videos_fld, v) for v in os.listdir(videos_fld)  if ".avi" in v]
# print(videos)
# dlc.create_new_project("ants-dlc", "federico", videos, working_directory=fld, copy_videos=True) # ? edit this

# %%
# ? config file
config_file =  os.path.join(fld, "config.yaml")
if not os.path.isfile(config_file):
    raise FileExistsError(config_file)


    
# %%
# ? Add vids to proj
# vids_to_add = ["D:\\Dropbox (UCL - SWC)\\Rotation_vte\\ants\\vids\\Decision Making Videos\\140R60L\\Ant3_Right_Crop30fps.avi",
#                "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\ants\\vids\\Decision Making Videos\\140R60L\\Ant2_Right_Crop30fps.avi",
#                "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\ants\\vids\\Decision Making Videos\\140R60L\\Ant1_Right_Crop30fps.avi"]
# dlc.add_new_videos(config_file, vids_to_add, copy_videos=True)

# %% 
# ? extract frames and label
# dlc.extract_frames(config_file, mode="manual")
# dlc.label_frames(config_file)

# %%
# ? Check labels
# dlc.check_labels(config_file)

# %%
# ? Create dataset
# dlc.create_training_dataset(config_file)
# 
# %%
# ? train network
# dlc.train_network(config_file)

# %%
# # ? Analyse
folds_to_analyse = ["100L60R", "100R60L", "120L60R", "120R60L", "140L60R", "140R60L"]
for f in folds_to_analyse:
    analysis_fld = os.path.join(videos_fld, "Decision Making Videos", f)
    videos = [os.path.join(analysis_fld, v) for v in os.listdir(analysis_fld) if ".avi" in v]
    dlc.analyze_videos(config_file, videos=videos, videotype="avi")
    dlc.create_labeled_video(config_file, videos, trailpoints=7, draw_skeleton=False)



#%%
# ? outliers
# dlc.extract_outlier_frames(config_file, videos)


#%%
# dlc.refine_labels(config_file)

#%%
