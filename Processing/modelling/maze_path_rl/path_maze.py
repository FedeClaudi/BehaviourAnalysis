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

		self.subgoals_fld = "Processing/modelling/maze_path_rl/subgoals"

		self.actions()
		

		self.free_states = free_states
		self.goal = goal
		self.maze = np.zeros((grid_size,grid_size))
		for i in self.free_states:
			self.maze[i[0]][i[1]] = 1

		self.maze_image = np.rot90(self.maze[::-1, :], 3)

		self.load_subgoals()

		self.intermediate_reached_subgoal = False

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
			self.subgoals = json.load(open(filename))
			# self.subgoals = [x[::-1] for x in temp]  # switch x and y because of opencv
		else:
			self.subgoals = self.get_subgoals()
			fixed = [x[::-1] for x in self.subgoals]
			self.subgoals = fixed
			with open(filename, "w") as f:
				f.write(json.dumps(self.subgoals))




	def reset(self,):
		# reset the environment
		if not self.randomise_start:
			self.start_index = self._start_index  #  always start at the same position
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
		for action in self.actions.values():
			action_next = self.move(action, curr)
			if action_next in self.free_states:
				legals.append(action)

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
		if action == "still":
			nxt_state = change(curr, 0, 0)
		elif action == "left":
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
			self.reward = 0


		if(self.next_state == self.goal):
			self.reward = 1
			self.game_over = True
		else:
			self.game_over = False

		return self.next_state, self.reward, self.game_over



	def get_geodesic_representation(self, remove_free_states = None):
		from sklearn import manifold

		# get a reduced list of freestates
		if remove_free_states is not None:
	
			free = [fs for fs in self.free_states if fs not in remove_free_states]
		else:
			free = self.free_states

		# X is a 2d array with shape: number-of-points by 2 [XY coordinates for each pixel on the maze]
		X = np.vstack(free)  # self.freestates is a list of points which are the pixels on the maze
		start_idx = free.index(self.start)
		goal_idx = free.index(self.goal)

		iso = manifold.Isomap(n_neighbors=6, n_components=2)  # fit the isomap
		iso.fit(X)

		# Make an image where each maze-pixel is colored accordingly to the distance from either the goal or the start
		idxs = [start_idx, goal_idx]
		titles = ['Geodesic distance from start', 'Geodesic distance from Shelter']
		
		self.geodesic_distance_states = []
		f, axarr = plt.subplots(ncols=2)
		for ax, target, title in zip(axarr, idxs, titles):
			m = np.full(self.maze_image.shape, np.nan)
			for n, coord in enumerate(free):
				d = iso.dist_matrix_[n, target]
				m[coord[1], coord[0]] = d

				if target == goal_idx:
					self.geodesic_distance_states.append(d)

			ax.imshow(m)
			ax.set(title=title)

		f.savefig("Processing/modelling/maze_path_rl/results/{}_geodesic.png".format(self.name))

