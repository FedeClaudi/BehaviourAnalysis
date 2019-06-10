import deeplabcut as dlc
import os

fld = r"D:\Dropbox (UCL - SWC)\Rotation_vte\DLC_nets\Egzona\DeepLabCut"

vid = r"O:\test.mp4"

os.chdir(fld)

dlc.check_labels("configEgzona.yaml")

# dlc.create_training_dataset("configEgzona.yaml")
# dlc.train_network("configEgzona.yaml", gputouse=0)

# dlc.analyze_videos("configEgzona.yaml", [vid])
# dlc.create_labeled_video("configEgzona.yaml", [vid])
