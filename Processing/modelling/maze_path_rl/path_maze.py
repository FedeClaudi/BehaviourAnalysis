import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from Processing.tracking_stats.math_utils import calc_distance_between_points_2d as dist
from Processing.tracking_stats.math_utils import geodist
import json



def get_maze_from_image(size, maze_design):
	# Load the model image
	folder = "Processing/modelling/maze_path_rl/mods"
	model = cv2.imread(os.path.join(folder, maze_design))

	# blur to remove imperfections
	kernel_size = 5
	model = cv2.blur(model, (kernel_size, kernel_size))

	# resize and change color space
	model = cv2.resize(model, (size, size))
	model = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)

	# threshold and rotate
	ret, model = cv2.threshold(model, 50, 255, cv2.THRESH_BINARY)
	model = np.rot90(model, 3)

	# return list of free spaces
	wh = np.where(model == 255)
	return [[x, y] for x, y in zip(wh[0], wh[1])]



class Maze(object):
	def __init__(self, name, grid_size, free_states, goal, start_position, start_index, randomise_start):
		self.name = name.split('.')[0]
		self._start_index = start_index
		self.start = start_position
		self.grid_size = grid_size
		self.randomise_start = randomise_start

		self.actions()
		
		self.free_states = free_states
		self.goal = goal
		self.maze = np.zeros((grid_size,grid_size))
		for i in self.free_states:
			self.maze[i[0]][i[1]] = 1

		self.maze_image = np.rot90(self.maze[::-1, :], 3)

		self.geodesic_distance, self.geodesic_gradient = geodist(self.maze, self.goal)

	def reset(self,):
		# reset the environment
		if not self.randomise_start:
			# self.start_index = self._start_index  #  always start at the same position
			self.curr_state = self.start.copy()
		else:
			self.start_index = np.random.randint(0,len(self.free_states))
			self.curr_state = self.free_states[self.start_index]

	def state(self):
		return self.curr_state

	def actions(self):
		self.actions = {
			0: 'left',
			1: 'right',
			2: 'up',
			3: 'down',
			4: "up-left",
			5: "up-right",
			6: "down-right",
			7: "down-left"
		}

	def get_available_moves(self, curr):
		"""[Get legal moves given the current position, i.e. moves that lead to a free cell]
		
		Arguments:
			curr {[list]} -- [x,y coordinates of the agent]
		"""
		legals = []

		surroundings = self.maze_image[curr[1]-1:curr[1]+2, curr[0]-1:curr[0]+2]

		actions = ["up-left", "up", "up-right", "left", "still", "right", "down-left", "down", "down-right"] # ! dont delete still from this list

		legals = [a for i, a in enumerate(actions) if surroundings.flatten()[i] and a != "still"]
		if not legals: raise ValueError
		return legals


	@staticmethod
	def move(action, curr):
		"""
			moves from current position to next position deoending on the action taken
		"""

		def change(current, x, y):
			"""
				change the position by x,y offsets
			"""
			return [current[0] + x, current[1] + y]

		# depending on the action taken, move the agent by x,y offsets
		up, down = -1, 1
		left, right = -1, 1

		if action == "left":
			nxt_state = change(curr, left, 0)
		elif action == "right":
			nxt_state = change(curr, right, 0)
		elif action == "up":
			nxt_state = change(curr, 0, up)
		elif action == "down":
			nxt_state = change(curr, 0, down)
		elif action == "up-left":
			nxt_state = change(curr, left, up)
		elif action == "up-right":
			nxt_state = change(curr, right, up)
		elif action == "down-right":
			nxt_state = change(curr, right, down)
		elif action == "down-left":
			nxt_state = change(curr, left, down)
		return nxt_state

	def act(self, action):
		# Move
		action_name = self.actions[action]
		self.next_state = self.move(action_name, self.curr_state)

		"""
			EVALUATE THE CONSEQUENCES OF MOVING
		"""
		if self.next_state in self.free_states:
			self.curr_state = self.next_state
			self.reward = 0
		else: 
			self.next_state = self.curr_state
			self.reward = -1 

		if(self.next_state == self.goal):
			self.reward = 1
			self.game_over = True
		else:
			self.game_over = False

		return self.next_state, self.reward, self.game_over 

	def ga_act(self, action):
		# Different set of rewards and punishment for GA evolution
		# Move
		action_name = self.actions[action]
		self.next_state = self.move(action_name, self.curr_state)

		"""
			EVALUATE THE CONSEQUENCES OF MOVING
		"""
		if self.next_state in self.free_states:
			self.curr_state = self.next_state
			self.reward = 0
		else: 
			self.next_state = self.curr_state
			self.reward = -1 # punish going against walls   

		if(self.next_state == self.goal):
			self.reward = 200  # reward getting to the goal
			self.game_over = True
		else:
			self.game_over = False

		return self.next_state, self.reward, self.game_over 

