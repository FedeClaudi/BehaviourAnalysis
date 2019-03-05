import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from Processing.tracking_stats.math_utils import calc_distance_between_points_2d as dist
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

		self.subgoals_fld = "Processing\modelling\maze_path_rl\subgoals"

		self.actions()
		

		self.free_states = free_states
		self.goal = goal
		self.maze = np.zeros((grid_size,grid_size))
		for i in self.free_states:
			self.maze[i[0]][i[1]] = 1

		self.load_subgoals()

	def get_subgoals(self):
		def register_click(event,x,y, flags, data):
			if event == cv2.EVENT_LBUTTONDOWN:
					# clicks = np.reshape(np.array([x, y]),(1,2))
					data.append([x,y])

		clicks_data = []

		maze = self.maze.copy()
		cv2.startWindowThread()
		cv2.namedWindow('background')
		cv2.imshow('background', maze)
		cv2.setMouseCallback('background', register_click, clicks_data)  # Mouse callback

		while True:
			k = cv2.waitKey(10)
			if k == ord('u'):
					print('Updating')
					# Update positions
					for x,y in clicks_data:
						cv2.circle(maze, (x, y), 2, (255, 0, 0), 2)
						cv2.imshow('background', maze)
			elif k == ord('q'):
				break
		return clicks_data

	def load_subgoals(self):
		filename = os.path.join(self.subgoals_fld, self.name+'.json')
		if os.path.isfile(filename):
			temp = json.load(open(filename))
			self.subgoals = [x[::-1] for x in temp]  # switch x and y because of opencv
		else:
			self.subgoals = self.get_subgoals()
			fixed = [x[::-1] for x in self.subgoals]
			self.subgoals = fixed
			with open(filename, "w") as f:
				f.write(json.dumps(self.subgoals))




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

	def act(self, action, move_away_reward, policy, goal):
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
		if policy == "away":
			curr_dist = dist(self.curr_state, self.start)
			next_dist = dist(self.next_state, self.start)
		elif policy == "direct_vector":
			curr_dist = dist(self.curr_state, goal)
			next_dist = dist(self.next_state, goal)
		elif policy == "intermediate":
			# get the distance to each subgoal
			subgoals_d = [dist(self.next_state, sg) for sg in self.subgoals]
			start_shelter_distance = dist(self.start, self.goal)
			subgoals_check = [d/start_shelter_distance for d in subgoals_d if d <= 4]
		elif policy == "combined":
			if dist(self.curr_state, self.start) < dist(self.next_state, self.start):  # check if moving away from shetler
				moved_away_check = True
			else:
				moved_away_check = False
			
			if dist(self.curr_state, goal) > dist(self.next_state, goal): # check if moving towards goal
				moved_towards_check = True
			else:
				moved_towards_check = True

		"""
			EVALUATE THE CONSEQUENCES OF MOVING
		"""
		# Consequences of moving
		if self.next_state in self.free_states:
			self.curr_state = self.next_state

			if policy == "away":
				if curr_dist > next_dist: # moving back towards start, not good 
					self.reward = - move_away_reward
				elif curr_dist < next_dist: # moving away from start
					self.reward = move_away_reward
				else:
					self.reward = 0

			elif policy == "direct_vector":
				if curr_dist > next_dist:  # moving towards goal
					self.reward = move_away_reward
				elif curr_dist < next_dist:  # moving away from goal
					self.reward = -move_away_reward
				else:
					self.reward = 0

			elif policy == "intermediate":
				if not subgoals_check:
					self.reward = 0
				else:
					self.reward = subgoals_check[0]*.5
					
			elif policy == "combined":
				self.reward = 0

				if moved_away_check: self.reward += .05
				else: self.reward -= .05

				if moved_towards_check: self.reward += .1
				else: self.reward -= .1

			else:
				self.reward = 0
		else: 
			self.next_state = self.curr_state
			self.reward = 0


		if(self.next_state == goal):
			self.reward += 1
			self.game_over = True
		else:
			self.game_over = False

		return self.next_state, self.reward, self.game_over
