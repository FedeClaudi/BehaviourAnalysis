import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from Processing.tracking_stats.math_utils import calc_distance_between_points_2d as dist

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
		self.name = name
		self._start_index = start_index
		self.start = start_position
		self.grid_size = grid_size
		self.num_actions = 4  
		self.actions()
		self.randomise_start = randomise_start
		# four actions in each state -- up, right, bottom, left

		self.free_states = free_states
		self.goal = goal
		self.maze = np.zeros((grid_size,grid_size))
		for i in self.free_states:
			self.maze[i[0]][i[1]] = 1

	def reset(self, random_start):
		# reset the environment

		if not self.randomise_start and random_start:
			self.start_index = self._start_index
		else:
			self.start_index = np.random.randint(0,len(self.free_states))
		self.curr_state = self.free_states[self.start_index]

	def state(self):
		return self.curr_state

	def actions(self):
		self.actions = {
			-1:'still',
			0: 'left',
			1: 'right',
			2: 'up',
			3: 'down',}

		# 	4: "up-left",
		# 	5: "up-right",
		# 	6: "down-right",
		# 	7: "down-left"
		# }

	def act(self, action, move_away_reward, goal):
		def move(action, curr):
			def change(current, x, y):
				return [current[0] + x, current[1] + y]

			if action == "still":
				nxt_state = change(curr, 0, 0)
			elif action == "left":
				nxt_state = change(curr, -1, 0)
			elif action == "right":
				nxt_state = change(curr, 1, 0)
			elif action == "up":
				nxt_state = change(curr, 0, 1)
			elif action == "down":
				nxt_state = change(curr, 0, -1)
			elif action == "up-left":
				nxt_state = change(curr, -1, 1)
			elif action == "up-right":
				nxt_state = change(curr, 1, 1)
			elif action == "down-right":
				nxt_state = change(curr, 1, -1)
			elif action == "down-left":
				nxt_state = change(curr, -1, -1)
			
			return nxt_state

		# Move
		action_name = self.actions[action]
		self.next_state = move(action_name, self.curr_state)

		# Calc distances
		if goal == "away":
			curr_dist = dist(self.curr_state, self.start)
			next_dist = dist(self.next_state, self.start)
		elif goal == "direct_vector":
			curr_dist = dist(self.curr_state, self.goal)
			next_dist = dist(self.next_state, self.goal)

		# Consequences of moving
		if self.next_state in self.free_states:
			self.curr_state = self.next_state

			if goal == "away":
				if curr_dist > next_dist: # moving back towards start, not good 
					self.reward = - move_away_reward
				elif curr_dist < next_dist: # moving away from start
					self.reward = move_away_reward
				else:
					self.reward = 0

			elif goal == "direct_vector":
				if curr_dist > next_dist:  # moving towards goal
					self.reward = move_away_reward
				elif curr_dist < next_dist:  # moving away from goal
					self.reward = -move_away_reward
				else:
					self.reward = 0

			else:
				self.reward = 0
		else: 
			self.next_state = self.curr_state
			self.reward = 0


		if(self.next_state == self.goal):
			self.reward = 1
			self.game_over = True
		else:
			if "away" in goal: self.reward -= .1
			self.game_over = False

		return self.next_state, self.reward, self.game_over
