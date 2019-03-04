from maze import Maze
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from generate_maze import get_maze_from_image
import seaborn as sns
from learner import Model

FLAG_policy = False
FLAG_showmaze = False

def plot_policy(model):
	Q = model.Q
	start = model.env.start
	free = model.env.free_states
	goal = model.env.goal
	grid_size = model.env.grid_size
	name = model.env.name


	f, axarr = plt.subplots(ncols=2, nrows=2)
	axarr= axarr.flatten()

	grid_size = len(Q)
	pol =  [[max(Q[i][j]) for i in range(grid_size)] for j in range(grid_size)]
	maze = [[1 if [i, j] in free else 0 for i in range(grid_size)] for j in range(grid_size)]
	act =  [[np.argmax(Q[j][i]) if pol[j][i] > 0 else np.nan for i in range(grid_size)] for j in range(grid_size)]

	axarr[0].imshow(maze, interpolation='none', cmap='gray')
	axarr[0].plot(start[0], start[1], 'o', color='g')
	axarr[0].plot(goal[0], goal[1], 'o', color='b')
	axarr[1].imshow(pol, interpolation='none', cmap='gray')

	axarr[2].imshow(act, cmap="tab20")

	axarr[3].imshow(maze, interpolation='none', cmap='gray')

	if model.walked is not None:
		axarr[3].scatter([x for (x,y) in model.walked], [y for (x,y) in model.walked], c='r', s=5, alpha=.8)


	axarr[0].set(title="maze")
	axarr[1].set(title="policy")
	axarr[2].set(title="actions")
	axarr[3].set(title="walk")

	for ax in axarr:
		ax.set(xticks=[], yticks=[])

	plt.savefig("policies/{}.png".format(name))
	

if __name__ == "__main__":

	print("Initializing")

	grid_size = 40
	

	# Define cells of the 2D matrix in which agent can move
	free_states = get_maze_from_image(grid_size, 0)
	
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
	env = Maze("asym-maze", grid_size, free_states, goal, start_position, start_index)
	# creating an instance of maze class

	print("Learning the policy")
	model = Model(env)
	model.train()
	model.walk()
	# model.plot_traces()

	print("Plotting the learned policy")
	plot_policy(model)
	# plot the action-value function 

	
print("Done! checkout task and policies folders")