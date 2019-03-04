import numpy as np
import matplotlib.pyplot as plt

class Maze(object):
	def __init__(self, name, grid_size, free_states, goal, start_position, start_index):
		self.name = name
		self._start_index = start_index
		self.start = start_position
		self.grid_size = grid_size
		self.num_actions = 4  
		self.actions()
		# four actions in each state -- up, right, bottom, left

		self.free_states = free_states
		self.goal = goal
		self.maze = np.zeros((grid_size,grid_size))
		for i in self.free_states:
			self.maze[i[0]][i[1]] = 1

	def reset(self):
		# reset the environment
		# self.start_index = np.random.randint(0,len(self.free_states))
		self.start_index = self._start_index
		self.curr_state = self.free_states[self.start_index]

	def state(self):
		return self.curr_state

	def draw(self, path = ""):
		# draw the maze configuration
		self.grid = np.zeros((self.grid_size, self.grid_size))
		for i in self.free_states:
			self.grid[i[1]][i[0]] = 0.5
		self.grid[self.goal[1]][self.goal[0]] = 1
		plt.figure(0)
		plt.clf()
		plt.imshow(self.grid, interpolation='none', cmap='gray')
		plt.savefig(path + "maze.png")

	def actions(self):
		self.actions = {
			-1:'still',
			0: 'left',
			1: 'right',
			2: 'up',
			3: 'down'

		}

	def act(self, action):
		# Move
		if(self.actions[action] == "still"):
			self.next_state = self.curr_state
		elif(self.actions[action] == "left"):
			self.next_state = [self.curr_state[0]-1,self.curr_state[1]]
		elif(self.actions[action] == "right"):
			self.next_state = [self.curr_state[0]+1,self.curr_state[1]]
		elif(self.actions[action] == "up"):
			self.next_state = [self.curr_state[0],self.curr_state[1]+1]
		elif(self.actions[action] == "down"):
			self.next_state = [self.curr_state[0],self.curr_state[1]-1]

		# Consequences of movign
		if self.next_state in self.free_states:
			self.curr_state = self.next_state
			self.reward = 0
		else:
			self.next_state = self.curr_state
			self.reward = 0

		if(self.next_state == self.goal):
			self.reward = 1
			self.game_over = True
		else:
			self.game_over = False

		return self.next_state, self.reward, self.game_over