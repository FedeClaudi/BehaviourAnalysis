import sys
sys.path.append('./')

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from random import choice
import pandas as pd
import random
from scipy.special import softmax
from tqdm import tqdm 
from mpl_toolkits.mplot3d import Axes3D
import yaml

from Processing.tracking_stats.math_utils import geodist

from Modelling.maze_solvers.agent import Agent


class GradientAgent(Agent):
	def __init__(self):
		Agent.__init__(self)

	def get_maze_options(self):
		self._reset()
		options = {}
	
		if self.maze_type == "modelbased":
			blocks = [["lambda", "beta1"], "alpha0", "alpha1", "beta0", "beta1", ["lambda", "beta0"],  ["lambda", "beta0"]]
			options_names = ["beta0", "alpha1", "alpha0", "dup1", "dup2", "beta1", "dup3"]
		elif self.maze_type == "asymmetric":
			blocks = ["none","right",] 
			options_names = ["right", "left"]   
		else:
			raise ValueError("unrecognised maze")

		
		
		f, axarr = plt.subplots(ncols =len(options_names))

		for i, (name, block) in enumerate(zip(options_names, blocks)):
			self.introduce_blockage(block)
			w = self.walk()
			self.plot_walk(w, ax=axarr[i])
			self._reset()

			if "dup" not in name:
				options[name] = w

		with open(self.options_file, 'a') as f:
			yaml.dump(options, f)

			

	def run(self):
		# prep figure
		f, axarr =  plt.subplots(ncols=6)

		# walk  on vanilla environment
		walk = self.walk()
		self.plot_walk(walk, background_image=self.geodesic_distance, color='r', ax=axarr[0])

		if self.maze_type == "modelbased":
			# Block LAMBDA and get the shortest path
			self.introduce_blockage('lambda')
			walk = self.walk()
			self.plot_walk(walk, background_image=self.geodesic_distance, color='r', ax=axarr[1])

			# reset the env
			self._reset()

			# Walk with closed LAMBDA but starting from I platform
			self.introduce_blockage("lambda")
			walk = self.walk(start='secondary')
			self.plot_walk(walk, background_image=self.geodesic_distance, color='r', ax=axarr[2])

			# ? Do a bi-phasic walk, 1st the agend doesnt know that L is closed and then it upates 
			self._reset()
			self.introduce_blockage("lambda", update=False)
			walk = self.walk()
			self.introduce_blockage("lambda")
			walk = self.walk(start=walk[-1], walk=walk)
			self.plot_walk(walk, background_image=self.geodesic_distance, color='r', ax=axarr[3])


			# Try blocking beta 0
			self._reset()
			self.introduce_blockage("beta0")
			walk = self.walk()
			self.plot_walk(walk, background_image=self.geodesic_distance, color='r', ax=axarr[4])

			# Try blocking beta 0 and lambda
			self._reset()
			self.introduce_blockage(["beta0", "lambda"])
			walk = self.walk()
			self.plot_walk(walk, background_image=self.geodesic_distance, color='r', ax=axarr[5])
		elif self.maze_type == "asymmetric":
			self.introduce_blockage("right")
			walk = self.walk()
			self.plot_walk(walk, background_image=self.geodesic_distance, color='r', ax=axarr[1])
		else:
			raise ValueError("unrecognised maze")


	def plot_geodesic_distance(self):
		f, ax = plt.subplots()
		ax.imshow(self.geodesic_distance)

		ax.scatter(self.curr_state[0], self.curr_state[1], c='g')
	
	def evaluate_geodesic_gradient(self, geodesic_map = None):
		'''     CALCULATE THE GRADIENT OF THE GEODESIC MAP AT THE CURRENT LOCATION      '''
		if geodesic_map is None: 
			geodesic_map = self.geodesic_distance
		
		surroundings = geodesic_map[self.curr_state[1]-1:self.curr_state[1]+2, self.curr_state[0]-1:self.curr_state[0]+2]
		return np.nanargmin(surroundings.flatten()), surroundings

		
	def step(self, current, geodesic_map = None):
		# Select which action to perform
		actions_complete = ["up-left", "up", "up-right", "left", "still", "right", "down-left", "down", "down-right"] # ! dont delete still from this list
		# ? this list includes the still action which is equivalent to the centre of the surrounding geodesic distance gradient

		action, surroundings = self.evaluate_geodesic_gradient(geodesic_map=geodesic_map)	
		action = actions_complete[action]


		# Move the agent accordingly
		return self.move(action, current)


	def walk(self, start=None, walk=None, geodesic_map=None, goal=None):
		"""[Creates a "gradient descent" walk between the starting point and the goal point, by default the goal]
		
		Keyword Arguments:
			start {[list]} -- [coordinates of start point or a name of start point] (default: {None})
			walk {[list]} -- [list of previous walk to append this walk to] (default: {None})
			geodesic_map {[list]} -- [list of geodesic distances from a specific place] (default: {None})
			goal {[list]} -- [goal location when walking to something] (default: {None})

		
		Returns:
			[type] -- [description]
		"""
		if walk is None: walk = []
		if start is None:
			curr = self.start_location.copy()
		else:
			if isinstance(start, str):
				curr = self.second_start_location.copy()
			else:
				curr = start
		self.curr_state = curr

		if goal is None: goal = self.goal_location

		step_n = 0
		# do each step
		while curr != goal and step_n < self.max_steps:
			step_n += 1
			walk.append(curr.copy())


			try:
				nxt = self.step(curr, geodesic_map=geodesic_map)
			except:
				break

			# Check that nxt is a legal move
			if nxt[0] < 0 or nxt[1] < 0 or nxt[0] > self.grid_size or nxt[1] > self.grid_size or nxt not in self.free_states:
				print("Terminating walk")
				break
			if nxt in self.free_states:
				curr = nxt
			self.curr_state = curr

		walk.append(curr)
		return walk


	def _reset(self):
		self.maze = self._maze.copy()
		self.geodesic_distance = self._geodesic_distance.copy()
		self.free_states  = self._free_states.copy()

	def introduce_blockage(self, bridge, update=True):
		blocks = self.get_blocked_states(bridge)

		for block in blocks:
			self.maze[block[1], block[0]] = 0 
			# self.maze[block[0], block[1]] = 0 
			self.geodesic_distance[block[1], block[0]] = np.nan # this will be overwritte if we are updating the geodesic distance


		# ? update the geodesic distance
		if update: self.geodesic_distance = geodist(self.maze, self.goal_location)
		

		

	def get_all_geo_distances(self):
		self.all_geo = np.zeros((len(self.free_states), len(self.free_states)))

		for row in np.arange(len(self.all_geo)):
			dist = self.geodist(self.maze, self.free_states[row])
			cleaned = [x for x in dist[~np.isnan(dist)].flatten()]


			self.all_geo[row, :] = cleaned
		
		
			
		# Check what the walks look like
		# draw 2 random states
		states = random.choices(np.arange(len(self.free_states)), k=2)
		geo = self.all_geo[states[1]]
		geo_img = self.create_maze_image_from_vals(geo)


		walk = self.walk(start=self.free_states[states[0]], goal=self.free_states[states[1]], geodesic_map=geo_img)

		f, ax = plt.subplots()
		self.plot_walk(walk, ax=ax)
		ax.scatter(self.free_states[states[0]][0], self.free_states[states[0]][1], c='g')
		ax.scatter(self.free_states[states[1]][0], self.free_states[states[1]][1], c='r')
		self.show()
		a = 1




if __name__ == "__main__":
	agent = GradientAgent()
	# agent.run()
	# agent.get_maze_options()

	agent.get_all_geo_distances()


	plt.show()
