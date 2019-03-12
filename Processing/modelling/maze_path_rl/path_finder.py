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
from Processing.modelling.maze_path_rl.path_agent import Model
from Processing.modelling.maze_path_rl.actor_critic import AA


FLAG_policy = False  
FLAG_showmaze = False

randomise_start_during_training = True
FLAG_load_trained = True


if __name__ == "__main__":

	print("Initializing")

	grid_size = 60

	maze_designs = [ "ModelBased.png", "PathInt.png", "PathInt2.png", "Square Maze.png"]
	
					#  "FourArms Maze.png", "TwoAndahalf Maze.png",
					# "Square Maze.png", "TwoArmsLong Maze.png", "mazemodel.png", "ModelBased.png", "ModelBased_mod.png"]
	
	for maze_design in maze_designs:
		print("\n\n\n", maze_design)

		# Define cells of the 2D matrix in which agent can move
		free_states = get_maze_from_image(grid_size, maze_design)
		
		# Define goal and start position
		goal = [round(grid_size/2), round(grid_size/4)]
		start_position = [round(grid_size/2), round(grid_size*.68)]	
		start_index = [i for i,e in enumerate(free_states) if e == start_position][0]
		


		# creating an instance of maze class
		# print("Creating Maze Environment")
		env = Maze(maze_design, grid_size, free_states,
					goal, start_position, start_index, randomise_start_during_training)
		
		# Train the Q-learning agent
		# print("Learning the policy")
		model = Model(env, FLAG_load_trained)
		model.train()
		model.save()

		# Get the shortest path to the shelter
		model.shortest_walk_f()

		# make random walks
		model.random_walks_f()

		# Find the alternative otpions
		model.find_options()

		# Make policy plot
		print("Plotting")
		model.policy_plot()

		# save model
		# model.save_model()

		# ? Actor critic part
		actor = AA(env.name)
		actor.train()

	plt.show()
