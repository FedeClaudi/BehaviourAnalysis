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

		self.value_est_iters = 10

		self.P = self.define_transitions_func()    			# transition function
		self.R = np.zeros(len(self.env.free_states))		# Reward function
		self.V = np.zeros(len(self.env.free_states))		# Value function

		self.n_steps = 1000

	def run(self):
		# prep figure
		f, axarr =  plt.subplots(ncols=4, nrows=2)

		# Learn the reward and transitions functions
		exploration = self.do_random_walk()

		# Perform value estimation
		self.value_estimation()
		self.plot_func("V", ax=axarr[0, 0], ttl="MB - Naive")

		# Do a walk and plot
		walk = self.walk_with_state_transitions()
		self.plot_walk(walk, ax=axarr[1, 0])

		# Introduce blocage on LAMBDA
		self.introduce_blockage('lambda')
		
		# Recompute values and do a walk
		self.reset_values()
		self.value_estimation()
		
		walk = self.walk_with_state_transitions()
		self.plot_func("V", ax=axarr[0, 1], ttl="Blocked LAMBDA")
		self.plot_walk(walk,ax=axarr[1, 1])

		# Relearn state transitions (reset to before blockage)
		self.reset()
		self.do_random_walk()

		# Block alpha and do a walk
		self.introduce_blockage("alpha0")
		self.reset_values()
		self.value_estimation()
		walk = self.walk_with_state_transitions()

		self.plot_func("V", ax=axarr[0, 2], ttl="Blocked - ALPHA0")
		self.plot_walk(walk, ax=axarr[1, 2])

		# Reset and repeat with both alphas closed
		self.reset()
		self.do_random_walk()
		self.introduce_blockage("alpha")
		self.reset_values()
		self.value_estimation()
		walk = self.walk_with_state_transitions()

		self.plot_func("V", ax=axarr[0, 3], ttl="Blocked - ALPHAs")
		self.plot_walk(walk, ax=axarr[1, 3])

		self.reset()

	def reset(self):
		self.env.free_states = self.free_states.copy()
		self.env.maze_image = self.maze_image.copy()

	def reset_values(self):
		self.V = np.zeros(len(self.env.free_states))	

	def plot_func(self, func="R", ax=None, ttl = None):
		if ax is None:
			f, ax = plt.subplots()

		if func == "R":
			policy = self.R
			title = ttl + " - reward function"
		else: 
			title = ttl + "- value function"
			policy = self.V

		# Create an image
		img = np.full(self.env.maze_image.shape, -1)
		for i, (x,y) in enumerate(self.env.free_states):
			img[x,y] = policy[i]

		# ax.scatter([x for x,y in self.env.free_states], [y for x,y in self.env.free_states], c=policy)
		ax.imshow(np.rot90(img, 3)[:, ::-1])
		ax.set(title = title, xticks=[], yticks=[])

	def define_transitions_func(self):
		return np.zeros((len(self.env.free_states), len(self.env.free_states), len(self.actions))).astype(np.int8)

	def do_random_walk(self):
		curr = self.env.start.copy()
		walk = []

		for step in np.arange(self.n_steps):
			walk.append(curr.copy())

			nxt, action_n = self.step("shelter", curr) # ? take random step

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

		if landing_states_values:

			if not np.max(landing_states_values) == 0:
				action_prob = softmax(landing_states_values)   # ? policy <- select highest value option with higher freq
			else:
				action_prob = [1/len(valid_actions) for i in np.arange(len(valid_actions))]

			value = np.sum([action_prob[i] * self.V[s1] for i, (s1,a) in enumerate(valid_actions)])
			self.V[idx] = reward + value
		

	def value_estimation(self, ax=None):
		print("\n\nValue estimation")
		for i in tqdm(range(self.value_est_iters)):
			for state in self.env.free_states:
				self.estimate_state_value(state)

		if ax is not None:
			ax.scatter([x for x,y in self.env.free_states], [y for x,y in self.env.free_states], c=self.V)
			

	def get_valid_actions(self, state):
		idx = self.get_state_index(state)
		valid_actions = []
		for state_idx, state in enumerate(self.P[idx]):
			[valid_actions.append((state_idx, action)) for action, p in enumerate(state) if p > 0]
		return valid_actions

	def walk_with_state_transitions(self):
		curr = self.env.start.copy()
		walk = []

		for step in np.arange(100):
			walk.append(curr)

			current_index = self.get_state_index(curr)
			valid_actions = self.get_valid_actions(curr)
			values = [self.V[si] for si,a in valid_actions]

			# softmax_choice = random.choices(valid_actions, k=1, weights=softmax(values))[0]
			# selected = self.env.free_states[softmax_choice[0]]

			selected = self.env.free_states[valid_actions[np.argmax(values)][0]]
			curr = selected

			if curr == self.env.goal: break
		walk.append(curr)
		return walk

	def mental_simulations(self, ttl, n_walks = 1):
		walks = [self.walk_with_state_transitions() for i in np.arange(n_walks)]
		min_len = np.min([len(w) for w in walks])
		shortest = [w for w in walks if len(w)==min_len][0]

		self.plot_walk(shortest, "MB - "+ttl)

	def plot_transitions_func(self):
		ncols = self.P.shape[-1]
		f, axarr = plt.subplots(ncols = 3, nrows=3)
		axarr = axarr.flatten()
		for i, ax in zip(np.arange(ncols), axarr):
			ax.imshow(self.P[:, :, i])
			ax.set(title=list(self.actions.keys())[i])

	def step(self, policy, current):
		legal_actions = [a for a in self.env.get_available_moves(current) if a != "still" not in a]
		action = choice(legal_actions)

		# move
		action_n = [k for k,v in self.actions.items() if v == action][0]
		return self.enact_step(current, action), action_n

	def introduce_blockage(self, bridge):
		if 'lambda' in bridge: blocks = self.states_to_block_mb_lambda
		elif bridge=='alpha1': blocks = self.states_to_block_mb_alpha1
		elif bridge=='alpha0': blocks = self.states_to_block_mb_alpha0
		else:
			blocks = self.states_to_block_mb_alpha1
			blocks.extend(self.states_to_block_mb_alpha0)


		for state in blocks:
			try:
				idx = self.get_state_index(state)
			except:
				pass
			else:
				self.P[:, idx] = 0

			self.env.maze_image[state[1], state[0]] = 0 
			




