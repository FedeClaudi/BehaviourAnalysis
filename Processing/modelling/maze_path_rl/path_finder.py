import sys
sys.path.append('./')

from copy import deepcopy
import numpy as np
import PyQt5

from tqdm import tqdm

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from Processing.modelling.maze_path_rl.path_maze import Maze, get_maze_from_image
from Processing.modelling.maze_path_rl.path_agent import Model
from Processing.modelling.maze_path_rl.actor_critic import AA
from Processing.modelling.maze_path_rl.tdrl import TDRL
from Processing.modelling.maze_path_rl.mbrl import MBRL


randomise_start_during_training = False
run_model_free = True

run_model_based = True


def generate_environment(maze_design, grid_size):
	print("\n\n\n", maze_design)
	# Define cells of the 2D matrix in which agent can move
	free_states = get_maze_from_image(grid_size, maze_design)
	print("   # free states:", len(free_states))
	
	# Define goal and start position
	# goal = [round(grid_size/2), round(grid_size/4)]
	goal = [10, 2]

	# start_position = [round(grid_size/2), round(grid_size*.85)]	
	start_position = [9, 14]
	start_index = [i for i,e in enumerate(free_states) if e == start_position][0]
	start_index = 0
	
	# creating an instance of maze class
	print("Creating Maze Environment")
	env = Maze(maze_design, grid_size, free_states,
				goal, start_position, start_index, randomise_start_during_training) 


	# plt.figure()
	# plt.imshow(env.maze_image)
	# plt.show()

	return env  


if __name__ == "__main__":

	print("Initializing")
	grid_size =  20
	maze_design = "ModelBased2.png"
	closed_maze_design = "ModelBased_mod.png"
	
	env = generate_environment(maze_design, grid_size)
	# env.get_geodesic_representation()

	if run_model_free:
		model = TDRL(env)
		model.run()

	if run_model_based:
		model = MBRL(env)
		model.run()
		



	plt.show()
	



