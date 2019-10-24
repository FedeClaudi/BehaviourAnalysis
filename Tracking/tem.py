import deeplabcut
import os

test = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\DLC_nets\\Nets\\DecisionMaze-Federico-2019-10-15\\config.yaml"
vidsf = "Z:\\branco\\Federico\\raw_behaviour\\maze\\video"
dest = "Z:\\branco\\Federico\\raw_behaviour\\maze\\newpose"

pose_files = [f.split("DLC")[0] for f in os.listdir(dest)]
vids = [os.path.join(vidsf, f) for f in os.listdir(vidsf) if ".mp4" in f and f.split(".")[0] not in pose_files and "threat" not in f.lower()]

print("{} vids".format(len(vids)))

not_processed = []
for video in vids[::-1]:
    # try:
    deeplabcut.analyze_videos(test, [video], videotype="mp4", destfolder=dest, save_as_csv=False, dynamic=(False, 0.5, 10))
    # except:
    #     print("\n\n\n\n\n\n COULD NOT PROCESS A VIDEO: {} \n\n\n\n\n\n\n\n".format(video))
        # not_processed.append(video)

print("\n\n Could not process these vids: {}".format(not_processed))