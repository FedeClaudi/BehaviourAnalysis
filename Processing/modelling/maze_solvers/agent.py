import sys
sys.path.append('./')
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import pandas as pd
from scipy.special import softmax
from tqdm import tqdm


from Processing.tracking_stats.math_utils import calc_distance_between_points_in_a_vector_2d as dist
from Processing.tracking_stats.math_utils import calc_distance_between_points_2d,  get_n_colors, calc_angle_between_points_of_vector, calc_ang_velocity, line_smoother

from Processing.modelling.maze_solvers.environment import Environment


class Agent(Environment):
	"""[General purpose agent class which contains core modules and info about the environment, it's subclassed by each individual model.
		Itself is a subclass of the environment class which defines the maze and so on. ]
	"""
	
	def __init__(self):
		Environment.__init__(self)

		# Max number of steps when walking
		self.max_steps = round(self.grid_size**2 / 2)

		# make a copy of free states and maze
		self._free_states = self.free_states.copy()
		self._maze = self.maze.copy()
		self._geodesic_distance = self.geodesic_distance.copy()



	def plot_walks(self, walks):
		f,ax = plt.subplots()
		ax.imshow(self.maze, cmap="Greys_r")
		for walk in walks:
			ax.scatter(np.vstack(walk)[:, 0], np.vstack(walk)[:, 1], alpha=.3,  s=150)

	def plot_walk(self, walk, ax=None, background=True, multiple=False, title=None, blocked=None, background_image=None, color=None):
		if ax is None:
			f,ax = plt.subplots()

		if background: 
			if background_image is None:
				ax.imshow(self.maze, cmap="Greys_r")
			else:
				ax.imshow(background_image)
			

		if not multiple:
			if color is None: color = np.arange(len(walk))
			ax.scatter(np.vstack(walk)[:, 0], np.vstack(walk)[:, 1], c = color, s=75)
		else:
			ax.scatter(np.vstack(walk)[:, 0], np.vstack(walk)[:, 1], alpha=.4, s=100)

		if title is None:
			ax.set(xticks=[], yticks=[])
		else:
			ax.set(title=title, xticks=[], yticks=[])


	def get_state_index(self, state):
		return [i for i,f in enumerate(self.free_states) if f == state][0]

	def state(self):
		return self.curr_state

	def get_blocked_states(self, bridge):
		if isinstance(bridge, list):
			blocks = []
			for br in bridge:
				blocks.extend(self.bridges_block_states[br])
		else:
			blocks = self.bridges_block_states[bridge]
		
		return blocks






