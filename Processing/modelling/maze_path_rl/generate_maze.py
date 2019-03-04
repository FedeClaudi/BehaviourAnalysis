import numpy as np
import cv2 



def get_maze_from_image(size, maze_n):
    # models = ["PathInt2.png", "Square Maze.png", "mazemodel.png"]
    models = ["mods/PathInt2.png"]
    model = cv2.resize(cv2.imread(models[maze_n]), (size, size))
    model = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)

    kernel_size = 2
    model = cv2.blur(model,(kernel_size, kernel_size))

    ret, model = cv2.threshold(model, 50, 255, cv2.THRESH_BINARY)
    # model = model[::-1, :]    
    model = np.rot90(model, 3)

    wh = np.where(model == 255)
    return [[x, y] for x,y in zip(wh[0], wh[1])]

    # cv2.imshow("m", model)
    # cv2.waitKey(1000)

