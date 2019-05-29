import sys
sys.path.append("./")


from Utilities.imports import *


def  run():
    source = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\ants\\dlc_vids"
    dest = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\ants\\dlc_vids_edited"

    editor = Editor()

    for vid in os.listdir(source):
        if vid in os.listdir(dest): continue

        # Open the video, convert to gray and rescale
        editor.compress_clip(os.path.join(source, vid), save_path = os.path.join(dest, vid),
                                compress_factor = .5)




if __name__ == "__main__":
    run()

