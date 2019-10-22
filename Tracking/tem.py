import deeplabcut
import os

cfg = "W:\\branco\\Federico\\raw_behaviour\\maze\\DLC_nets\\DecisionMaze-Federico-2019-10-15\\config.yaml"
vidsf = "W:\\branco\\Federico\\raw_behaviour\\maze\\video"
dest = "W:\\branco\\Federico\\raw_behaviour\\maze\\newpose"

vids = [os.path.join(vidsf, f) for f in os.listdir(vidsf) if ".mp4" in f]

print("{} vids".format(len(vids)))

not_processed = []
for video in vids:
    try:
        deeplabcut.analyze_videos(cfg, [video], videotype="avi", destfolder=dest, save_as_csv=False, dynamic=(False, 0.5, 10))
    except:
        print("\n\n\n\n\n\n COULD NOT PROCESS A VIDEO: {} \n\n\n\n\n\n\n\n".format(video))
        not_processed.append(video)

print("\n\n Could not process these vids: {}".format(not_processed))