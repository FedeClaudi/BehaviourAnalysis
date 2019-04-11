import sys
sys.path.append('./')

from copy import deepcopy
import numpy as np
import PyQt5

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from Processing.modelling.maze_path_rl.path_maze import Maze, get_maze_from_image
from Processing.modelling.maze_path_rl.path_agent import Model
from Processing.modelling.maze_path_rl.actor_critic import AA
from Processing.modelling.maze_path_rl.tdrl import TDRL
from Processing.modelling.maze_path_rl.mbrl import MBRL

FLAG_policy = False  
FLAG_showmaze = False

randomise_start_during_training = True
FLAG_load_trained = True


def generate_environment(maze_design, grid_size):
	print("\n\n\n", maze_design)
	# Define cells of the 2D matrix in which agent can move
	free_states = get_maze_from_image(grid_size, maze_design)
	print("   # free states:", len(free_states))
	
	# Define goal and start position
	goal = [round(grid_size/2), round(grid_size/4)]
	start_position = [round(grid_size/2), round(grid_size*.85)]	
	start_position = [9, 14]
	# try:
	start_index = [i for i,e in enumerate(free_states) if e == start_position][0]
	# except: 
	# start_index = 0
	
	# creating an instance of maze class
	# print("Creating Maze Environment")
	env = Maze(maze_design, grid_size, free_states,
				goal, start_position, start_index, randomise_start_during_training)

	return env


if __name__ == "__main__":

	print("Initializing")

	grid_size = 20

	maze_design = "PathInt2.png"
	closed_maze_design = "PathInt2_closed.png"


	
	env = generate_environment(maze_design, grid_size)
	# env.get_geodesic_representation()


	model = MBRL(env)
	model.do_random_walk()

	for i in range(50):
		model.value_estimation()

	model.mental_simulations('naive')


	model.introduce_blockage()

	f, ax = plt.subplots()
	for i in range(50):
		model.value_estimation()

	model.mental_simulations('blocked')


	# Train the Q-learning agent
	model = TDRL(env)
	model.train()
	model.shortest_walk_f()

	# change environment to closed one
	env = generate_environment(closed_maze_design, grid_size)
	model.env = env

	model.train()
	model.shortest_walk_f()

	plt.show()
	











	# model.save()

	# Get the shortest path using geodesic distances
	model.geodesic_walk()

	# # Get the shortest path to the shelter
	# print("finding options")
	# model.shortest_walk_f()

	# # make random walks
	# model.random_walks_f()

	# # Find the alternative otpions
	# model.find_options()

	# Make policy plot
	# print("Plotting")
	model.policy_plot()

	# # save model
	model.save_model()

	# # ? Actor critic part
	# actor = AA(env.name)
	# actor.train()

plt.show()
