import sys
sys.path.append('./') 
# Add Philip's script to path
sys.path.append('C:\\Users\\Federico\\Documents\\GitHub\\CommonCoordinateBehaviour')
# THIS: https://github.com/BrancoLab/Common-Coordinate-Behaviour
try:
    from video_funcs import register_arena
except: pass
import os 
import numpy as np
import cv2

from Utilities.file_io.files_load_save import load_yaml

def run(videopath, maze_model=None):
    if maze_model is None:
        # Get the maze model template
        maze_model = cv2.imread('Utilities\\video_and_plotting\\mazemodel.png')
        maze_model = cv2.resize(maze_model, (1000, 1000))
        maze_model = cv2.cvtColor(maze_model,cv2.COLOR_RGB2GRAY)

    # Define points to be used for alignemt
    points = np.array([[435, 290], [565, 290], [435, 710], [565, 710]])

    # <- uncomment to display point on image
    # for i, p in enumerate(points):
    #     cv2.circle(maze_model, tuple(p), 7, (i*50, 125, 125), -1)
    # cv2.imshow('cc', maze_model)
    # cv2.waitKey(2000)
    # cv2.destroyAllWindows()


    # Get the background (first frame) of the video being processed
    try:
        cap = cv2.VideoCapture(videopath)
        if not cap.isOpened(): ValueError
    except:
        raise FileNotFoundError('Couldnt open file ', videopath)

    # Get video frame width and height and check if they are within limits
    width = int(cap.get(3))  
    height = int(cap.get(4))

    if width > 1000 or height > 1000:
        raise ValueError('Frame too large: ', width, height)
    
    # Pad the background frame to be of the right size for template matching
    ret, frame = cap.read()
    if not ret: 
        raise FileNotFoundError('Could not open videopath', videopath)
    top_pad, side_pad = int(np.floor((1000-height)/2)), int(np.floor((1000-width)/2))
    padded = cv2.copyMakeBorder(frame, top_pad,  top_pad, side_pad, side_pad,
                                cv2.BORDER_CONSTANT,value=[0, 0, 0])
    try:
        padded = cv2.cv2.cvtColor(padded,cv2.COLOR_RGB2GRAY)
    except:
        raise ValueError(frame.shape)

    if padded.shape != maze_model.shape:
        raise ValueError('Shapes dont match ', padded.shape, maze_model.shape)
    if padded.dtype != maze_model.dtype:
        raise ValueError('Datatypes dont match', padded.dtype, maze_model.dtype)

    # Get fisheye correction matriz path
    fisheye = 'Utilities\\video_and_plotting\\fisheye_maps.npy'

    # Call the registration function
    """
        Credit to Philip Shamas (Branco Lab) -  https://github.com/BrancoLab/Common-Coordinate-Behaviour
    """
    paths = load_yaml('paths.yml')
    savepath = paths['commoncoordinatebehaviourmatrices']
    savename = os.path.split(videopath)[-1]
    registration = register_arena(padded, 'nofisheye', 0, 0, maze_model, points, savepath, savename)

    return registration[0], points, top_pad, side_pad


def define_roi_on_template():
    maze_model = cv2.imread('Utilities\\video_and_plotting\\mazemodel.png')
    if maze_model is None:
        maze_model = cv2.imread('Utilities/video_and_plotting/mazemodel.png')
    maze_model = cv2.resize(maze_model, (1000, 1000))
    maze_model = cv2.cvtColor(maze_model, cv2.COLOR_RGB2GRAY)
    r = cv2.selectROI(maze_model)
    print('\n\nSelect ROI:\n    x - {}, width - {}\n    y - {}, hieght - {}'.format(
                                    r[1], r[3], r[0], r[2]))
    # cv2.imshow('mm', maze_model)
    # while True:
    #     cv2.waitKey(1)

if __name__ == "__main__":
    testfile = 'Z:\\branco\\Federico\\raw_behaviour\\maze\\video\\180606_CA2762.avi'
    # run(testfile)
    define_roi_on_template()
    





