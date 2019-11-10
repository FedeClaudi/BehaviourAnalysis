import deeplabcut
import os

cfg = "W:\\branco\\Federico\\raw_behaviour\\maze\\DLC_nets\\DecisionMaze-Federico-2019-10-15\\config.yaml"
vidsf = "W:\\branco\\Federico\\raw_behaviour\\maze\\video"
dest = "W:\\branco\\Federico\\raw_behaviour\\maze\\newpose"

pose_files = [f.split("DLC")[0] for f in os.listdir(dest)]
vids = [os.path.join(vidsf, f) for f in os.listdir(vidsf) if ".avi" in f and f.split(".")[0] in pose_files and "threat" not in f.lower()]

print("{} vids".format(len(vids)))

# not_processed = []

deeplabcut.filterpredictions(cfg, vids, videotype="mp4", destfolder=dest, save_as_csv=False, filtertype='median')
