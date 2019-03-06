import sys
sys.path.append('./')

from copy import deepcopy
import numpy as np
import PyQt5

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from Processing.modelling.maze_path_rl.path_maze import Maze, get_maze_from_image
from Processing.modelling.maze_path_rl.path_learner import Model
from Processing.modelling.maze_path_rl.path_walker import Walker


FLAG_policy = False  
FLAG_showmaze = False

randomise_start_during_training = True
FLAG_load_trained = True


if __name__ == "__main__":

	print("Initializing")

	grid_size = 60

	maze_designs = ["PathInt.png", "PathInt2.png", "FourArms Maze.png", "TwoAndahalf Maze.png",
					"Square Maze.png", "TwoArmsLong Maze.png", "mazemodel.png", "ModelBased.png", "ModelBased_mod.png"]
	
	for maze_design in maze_designs:
		print("\n\n\n", maze_design)

		# Define cells of the 2D matrix in which agent can move
		free_states = get_maze_from_image(grid_size, maze_design)
		
		# Define goal and start position
		goal = [round(grid_size/2), round(grid_size/4)]
		start_position = [round(grid_size/2), round(grid_size*.68)]	
		start_index = [i for i,e in enumerate(free_states) if e == start_position][0]
		
		# Show the  maze as an image
		if FLAG_showmaze:
			maze_to_print = np.zeros((grid_size, grid_size)).astype(np.int16)
			for state in free_states: maze_to_print[state[0], state[1]] = 5
			maze_to_print[goal[0], goal[1]] = 9
			maze_to_print[start_position[0], start_position[1]] = 25
			
			plt.imshow(maze_to_print)
			plt.show()

		# creating an instance of maze class
		print("Creating Maze Environment")
		env = Maze(maze_design, grid_size, free_states,
					goal, start_position, start_index, randomise_start_during_training)
		

		print("Learning the policy")
		model = Model(env, FLAG_load_trained)
		model.train()

		# walk
		walker = Walker(model)
		walker.walk()

		# plot
		walker.policy_plot()
		walker.walks_plot()
		walker.subgoals_plot()

		# Save policies
		model.save()

	# plt.show()
