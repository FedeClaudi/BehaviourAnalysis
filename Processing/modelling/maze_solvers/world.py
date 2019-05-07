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
    def __init__(self):
        # define variables
        self.maze_design = "ModelBased2.png"
        self.maze_type = "modelbased"

        self.grid_size = 40

        self.randomise_start_location_during_training = False

        if self.maze_type == "modelbased":
            self.goal_location =  [20, 4] # [10, 2]  # for MBv2 like mazes
        elif self.maze_type == "asymmetric":
            self.goal_location = [19, 10] # [9, 5]     # for pathint2 like maze
        else:
            raise ValueError("unrecognised maze")
            
        self.start_location = [20, 32] # [9, 14]
        self.second_start_location = [19, 17] # [9, 9]  # alternative start

        # static vars
        self.maze_models_folder = "Processing\modelling\maze_solvers\mazes_images"

        

if __name__ == "__main__":
    w = World()
