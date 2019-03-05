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


FLAG_policy = False
FLAG_showmaze = False

randomise_start_during_training = True

def plot_policy(model):
	Q = model.Q
	start = model.env.start
	free = model.env.free_states
	goal = model.env.goal
	grid_size = model.env.grid_size
	name = model.env.name

	f, axarr = plt.subplots(ncols=3, nrows=len(list(model.Q.keys())))

	for i, (Q_name, Q) in enumerate(model.Q.items()):
		print(Q_name)
		grid_size = len(Q)
		pol =  [[max(Q[i][j]) for i in range(grid_size)] for j in range(grid_size)]
		maze = [[1 if [i, j] in free else 0 for i in range(grid_size)] for j in range(grid_size)]
		act =  [[np.argmax(Q[i][j]) if pol[j][i] > 0 else np.nan for i in range(grid_size)] for j in range(grid_size)]

		axarr[i, 0].imshow(maze, interpolation='none', cmap='gray')
		axarr[i, 0].plot(start[0], start[1], 'o', color='g')
		axarr[i, 0].plot(goal[0], goal[1], 'o', color='b')

		if model.walked is not None:
			vals = [0, .5, .1]
			for val, walked in zip(vals, model.walked[Q_name]):
				axarr[i, 0].scatter([x for (x, y) in walked],
								[y for (x, y) in walked], s=5, alpha=.4)
			# axarr[i, 0].legend()

		axarr[i, 1].imshow(act, interpolation='none',  cmap="tab20")
		axarr[i, 2].imshow(pol, interpolation='none')

		axarr[i, 0].set(title=list(model.Q.keys())[i])
		axarr[i, 1].set(title="actions")
		axarr[1, 2].set(title="Policy")


	axarr = axarr.flatten()
	for ax in axarr:
		ax.set(xticks=[], yticks=[])

	plt.savefig("Processing/modelling/maze_path_rl/results/{}.png".format(name))
	

if __name__ == "__main__":

	print("Initializing")

	grid_size = 60

	maze_designs = ["PathInt.png", "PathInt2.png", "FourArms Maze.png", "TwoAndahalf Maze.png",
					"Square Maze.png", "TwoArmsLong Maze.png", "mazemodel.png", "ModelBased.png", "ModelBased_mod.png"]
	for maze_design in maze_designs:
		print(maze_design)

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

		print("Creating Maze Environment")
		env = Maze(maze_design, grid_size, free_states,
		           goal, start_position, start_index, randomise_start_during_training)
		# creating an instance of maze class

		print("Learning the policy")
		model = Model(env)
		model.train()
		model.walk()
		# model.plot_traces()

		print("Plotting the learned policy")
		plot_policy(model)
		# plot the action-value function 

	
