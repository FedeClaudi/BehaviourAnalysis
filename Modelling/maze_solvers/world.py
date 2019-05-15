import sys
sys.path.append('./')

import numpy as np
from tqdm import tqdm
import cv2 
import os
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


"""
    This script collects all individual maze solvers and coordinates the creation of the environment and the usage of each agent
"""


class World:
    def __init__(self, grid_size=None, **kwargs):
        self.stride = 1 # by how much they can move at each step 

        # define variables
        self.maze_design = "PathInt2.png"
        self.maze_type = "asymmetric_large"

        if grid_size is not None:
            self.grid_size = grid_size
        else:
            self.grid_size = 40 # ? default value
        
        self.randomise_start_location_during_training = False

        if "modelbased" in self.maze_type:
            self.goal_location =  [20, 4] 
        elif "asymmetric" in self.maze_type:
            self.goal_location = [19, 10] 
        else:
            raise ValueError("unrecognised maze")
            
        self.start_location = [20, 32] # [9, 14]
        self.second_start_location = [19, 17] # [9, 9]  # alternative start

        # Check if other value were passed by the user
        for k,v in kwargs.items():
            if k == "start_loc": self.start_location = v
            elif k == "goal_loc": self.goal_location = v
            elif k == "stride": self.stride   = v

        # static vars
        self.maze_models_folder = "Modelling\maze_solvers\mazes_images"

        

if __name__ == "__main__":
    w = World()
