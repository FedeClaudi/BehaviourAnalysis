import sys
sys.path.append('./')
import matplotlib.pyplot as plt
import numpy as np
from Processing.tracking_stats.math_utils import calc_distance_between_points_in_a_vector_2d as dist
from Processing.tracking_stats.math_utils import calc_distance_between_points_2d as dist
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
from tqdm import tqdm 


from Processing.modelling.maze_path_rl.tdrl import Agent


class MBRL(Agent):
	def __init__(self, env):
		Agent.__init__(self, env)

		self.P = self.define_transitions_func()    			# transition function
		self.R = np.zeros(len(self.env.free_states))		# Reward function
		self.V = np.zeros(len(self.env.free_states))		# Value function

		self.n_steps = 1000

	
	def define_transitions_func(self):
		return np.zeros((len(self.env.free_states), len(self.env.free_states), len(self.actions))).astype(np.int8)

	def do_random_walk(self):
		curr = self.env.start.copy()
		walk = []

		for step in tqdm(np.arange(self.n_steps)):
			walk.append(curr.copy())

			nxt, action_n = self.step("shelter", curr, random_action=True)

			# update state transition function
			curr_index  = self.get_state_index(curr)
			nxt_index  = self.get_state_index(nxt)

			if nxt == self.env.goal:
				self.R[nxt_index] = 1

			self.P[curr_index, nxt_index, action_n] = 1  # set as 1 because we are in a deterministic world

			curr = nxt

		walk.append(curr)
		return walk
	
	def estimate_state_value(self, state):
		idx = self.get_state_index(state)
		reward = self.R[idx]

		valid_actions = self.get_valid_actions(state)

		landing_states_values = [self.V[si] for si, a in valid_actions]
		if not np.max(landing_states_values) == 0:
			action_prob = softmax(landing_states_values)   # ? policy <- select highest value option with higher freq
		else:
			action_prob = [1/len(valid_actions) for i in np.arange(len(valid_actions))]

		value = np.sum([action_prob[i] * self.V[s1] for i, (s1,a) in enumerate(valid_actions)])
		self.V[idx] = reward + value
		

	def value_estimation(self):
		for state in self.env.free_states:
			self.estimate_state_value(state)

		# ax.scatter([x for x,y in self.env.free_states], [y for x,y in self.env.free_states], c=self.V)
			

	def get_valid_actions(self, state):
		idx = self.get_state_index(state)
		valid_actions = []
		for state_idx, state in enumerate(self.P[idx]):
			[valid_actions.append((state_idx, action)) for action, p in enumerate(state) if p > 0]
		return valid_actions

	def walk_with_state_transitions(self):
		curr = self.env.start.copy()
		walk = []

		for step in tqdm(np.arange(100)):
			walk.append(curr)

			current_index = self.get_state_index(curr)
			valid_actions = self.get_valid_actions(curr)
			values = [self.V[si] for si,a in valid_actions]
			selected = self.env.free_states[valid_actions[np.argmax(values)][0]]

			# for each possible action check which one would bring you closer to the shelter
			# possibe_states_value = [self.R[a[0]] for a in valid_actions]
			# selected = self.env.free_states[valid_actions[np.argmin(possibe_states_value)][0]]

			# # select an option - those that have smaller distance select with higher prob. 
			# # future_states_probs = softmax(1 - softmax(possible_states_distances))
			# # selected = self.env.free_states[random.choices(valid_actions, k=1, weights=future_states_probs)[0][0]]

			curr = selected

			if curr == self.env.goal: break
		walk.append(curr)
		return walk

		# self.plot_walk(walk)

	def mental_simulations(self, ttl, n_walks = 10):
		walks = [self.walk_with_state_transitions() for i in np.arange(n_walks)]
		min_len = np.min([len(w) for w in walks])
		shortest = [w for w in walks if len(w)==min_len][0]

		self.plot_walk(shortest, "MB shortest walk - "+ttl)

		a = 1


	def plot_transitions_func(self):
		ncols = self.P.shape[-1]
		f, axarr = plt.subplots(ncols = 3, nrows=3)
		axarr = axarr.flatten()
		for i, ax in zip(np.arange(ncols), axarr):
			ax.imshow(self.P[:, :, i])
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

	def introduce_blockage(self):
		states_to_block = [[11, 8], [12, 8], [11, 7], [11, 8]]

		for state in states_to_block:
			idx = self.get_state_index(state)
			self.P[:, idx] = 0

			




