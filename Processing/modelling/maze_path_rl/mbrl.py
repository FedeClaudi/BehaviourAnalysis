import sys
sys.path.append('./')
import matplotlib.pyplot as plt
import numpy as np
from Processing.tracking_stats.math_utils import calc_distance_between_points_in_a_vector_2d as dist
from Processing.tracking_stats.math_utils import calc_distance_between_points_2d,  get_n_colors, calc_angle_between_points_of_vector, calc_ang_velocity, line_smoother
from math import exp  
import json
import os
from random import choice
import pandas as pd
# from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.signal import resample
import random
import pickle
from scipy.special import softmax
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import gaussian_filter


from Processing.modelling.maze_path_rl.tdrl import Agent


class MBRL(Agent):
	def __init__(self, env):
		Agent.__init__(self, env)

		self.transitions_func = self.define_transitions_func()
		self.Q = self.define_Q()

		self.n_steps = 10000

	
	def define_transitions_func(self):
		return np.zeros((len(self.env.free_states), len(self.env.free_states), len(self.actions))).astype(np.int8)

	def define_Q(self):
		return np.zeros((self.env.grid_size, self.env.grid_size, len(self.actions)))

	def do_random_walk(self):
		curr = self.env.start.copy()
		walk = []

		for step in np.arange(self.n_steps):
			if step % 1000 == 0: 
				print("step: ", step)
			walk.append(curr.copy())

			nxt, action_n = self.step("shelter", curr, random_action=True)

			# update state transition function
			curr_index  = self.get_state_index(curr)
			nxt_index  = self.get_state_index(nxt)

			self.transitions_func[curr_index, nxt_index, action_n] = 1  # set as 1 because we are in a deterministic world

			# get the dist between the current location and the next one
			distance = calc_distance_between_points_2d(curr, nxt)

			if distance < 1:
				print("didn't move at step: ", step)

			curr = nxt

		walk.append(curr)

		# self.plot_transitions_func()
		self.plot_walk(walk)
		return walk


	

	def walk_with_state_transitions(self):
		curr = self.env.start.copy()
		walk = []
		for step in np.arange(self.n_steps):
			if step % 1000 == 0: 
				print("step: ", step)
			current_index = self.get_state_index(curr)

			valid_actions = []
			for state_idx, state in enumerate(self.transitions_func[current_index]):
				[valid_actions.append((state_idx, action)) for action, p in enumerate(state) if p > 0]

			walk.append(curr)
			curr = self.env.free_states[random.choice(valid_actions)[0]]

		self.plot_walk(walk)




	def plot_transitions_func(self):
		ncols = self.transitions_func.shape[-1]
		f, axarr = plt.subplots(ncols = 3, nrows=3)
		axarr = axarr.flatten()
		for i, ax in zip(np.arange(ncols), axarr):
			ax.imshow(self.transitions_func[:, :, i])
			ax.set(title=list(self.actions.keys())[i])


	def step(self, policy, current, random_action = False):
		# Select which action to perform
		if not random_action:
			action_values = self.Q[policy][current[0]][current[1]]
			action = np.argmax(action_values)
			action = self.env.actions[action]
		else:
			legal_actions = [a for a in self.env.get_available_moves(current) if a != "still" not in a]
			action = choice(legal_actions)

		# move
		action_n = [k for k,v in self.actions.items() if v == action][0]
		return self.enact_step(current, action), action_n




