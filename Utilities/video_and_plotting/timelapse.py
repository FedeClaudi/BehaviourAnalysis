import sys
sys.path.append("./")

from Utilities.imports import *


def stitch(fld):
    print("stitching")
    editor = Editor()#
    for i, img in tqdm(enumerate(os.listdir(fld))):
        if i == 0:
            height, width, layers= cv2.imread(os.path.join(fld, img)).shape
            video= editor.open_cvwriter(os.path.join(fld, "video.mp4"), width, height, framerate = 10, iscolor=True)

        video.write(cv2.imread(os.path.join(fld, img)))
    video.release()
    print("done")



if __name__ == "__main__":
    main_fld = "D:\\Dropbox (UCL - SWC)\\Apps\\ants_uploader\\timelapses"
    sub_fld = "190606_timelapse"

    stitch(os.path.join(main_fld, sub_fld))


