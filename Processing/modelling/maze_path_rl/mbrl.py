import sys
sys.path.append('./')
import matplotlib.pyplot as plt
import numpy as np
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
from scipy.special import expit as sigmoid


from Processing.modelling.maze_path_rl.tdrl import Agent


class MBRL(Agent):
	def __init__(self, env):
		Agent.__init__(self, env)

		self.value_est_iters = 50

		self.P = self.define_transitions_func()    			# transition function
		self.R = np.zeros(len(self.env.free_states))		# Reward function
		self.V = np.zeros(len(self.env.free_states))		# Value function

		self.n_steps = 10000

	def run(self):
		# prep figure
		f, axarr =  plt.subplots(ncols=5, nrows=3)

		# Learn the reward and transitions functions
		exploration = self.explore()

		# Perform value estimation
		self.value_estimation()
		self.plot_func("P", ax=axarr[0, 0], ttl="MB - Naive")
		self.plot_func("V", ax=axarr[1, 0], ttl="")

		# Do a walk and plot
		walk = self.walk_with_state_transitions()
		self.plot_walk(walk, ax=axarr[2, 0])

		# Introduce blocage on LAMBDA
		self.introduce_blockage('lambda')
		
		# Recompute values and do a walk
		self.reset_values()
		self.value_estimation()
		
		walk = self.walk_with_state_transitions()
		self.plot_func("P", ax=axarr[0, 1], ttl="Blocked LAMBDA")
		self.plot_func("V", ax=axarr[1, 1], ttl="")
		self.plot_walk(walk,ax=axarr[2, 1])

		# Do one trial starting at P with LAMBDA *fully* closed
		self.introduce_blockage('lambda', p=0)

		self.reset_values()
		self.value_estimation()
		walk = self.walk_with_state_transitions(start="secondary")
		self.plot_func("P", ax=axarr[0, 4], ttl="Start at P")
		self.plot_func("V", ax=axarr[1, 4], ttl="")
		self.plot_walk(walk,ax=axarr[2, 4])


		# Relearn state transitions (reset to before blockage)
		self.reset()
		self.explore()

		# Block alpha and do a walk
		self.introduce_blockage("alpha1")
		self.reset_values()
		self.value_estimation()
		walk = self.walk_with_state_transitions()

		self.plot_func("P", ax=axarr[0, 2], ttl="Blocked - ALPHA1")
		self.plot_func("V", ax=axarr[1, 2], ttl="")
		self.plot_walk(walk, ax=axarr[2, 2])

		# Reset and repeat with both alphas closed
		self.reset()
		self.explore()
		self.introduce_blockage("alpha")
		self.reset_values()
		self.value_estimation()
		walk = self.walk_with_state_transitions()

		self.plot_func("P", ax=axarr[0, 3], ttl="Blocked - ALPHAs")
		self.plot_func("V", ax=axarr[1, 3], ttl="")
		self.plot_walk(walk, ax=axarr[2, 3])

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
		elif func=='V': 
			title = ttl + " - value function"
			policy = self.V
		else:
			title = ttl + " - transition function"
			policy = self.P.copy()
			policy[policy == 0] = np.nan
			# policy = np.sum(self.P, 1)[:, 0]
			policy = np.nanmin(np.nanmin(policy, 2),1)
		# Create an image
		img = np.full(self.env.maze_image.shape, np.nan)
		for i, (x,y) in enumerate(self.env.free_states):
			img[x,y] = policy[i]

		# ax.scatter([x for x,y in self.env.free_states], [y for x,y in self.env.free_states], c=policy)
		ax.imshow(np.rot90(img, 3)[:, ::-1])
		ax.set(title = title, xticks=[], yticks=[])

	def define_transitions_func(self):
		return np.zeros((len(self.env.free_states), len(self.env.free_states), len(self.actions))).astype(np.float16)

	def explore(self):
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
		# Get the reward at the current state
		idx = self.get_state_index(state)
		reward = self.R[idx]

		# Get which actions can be perfomed and the values of the states they lead to
		valid_actions = self.get_valid_actions(state)
		landing_states_values = [self.V[si] for si, a, p in valid_actions] #  get the values of the landing states to guide the prob of action selection
		if not np.any(landing_states_values): 
			action_probs = [1/len(valid_actions) for i in np.arange(len(valid_actions))]  # if landing states have no values choose each random with same prob
		else:
			action_probs = softmax(landing_states_values)

		"""
			landing_states_values = [self.V[si] for si, a, p in valid_actions]

			if landing_states_values:
				# If the landing states don't have values, choose a random one
				if not np.max(landing_states_values) == 0:
					action_prob = softmax(landing_states_values)   # ? policy <- select highest value option with higher freq
				else:
					# Each action has equal probability of bein selected
					action_prob = [1/len(valid_actions) for i in np.arange(len(valid_actions))]

				# The value of the current state is given by the sum for each action of the product of:
				# the probability of taking the action
				# the value of the landing state
				# the probaility of taking getting to the state (transtion function)

				value = np.sum([action_prob[i] * self.V[s1] * p for i, (s1,a, p) in enumerate(valid_actions)])
				self.V[idx] = reward + value
		"""
		value = 0 # initialise the value as 0
		if valid_actions:
			for (s1_idx, action, transition_prob), action_prob in zip(valid_actions, action_probs):
				r = self.R[s1_idx]  # reward at landing state
				# transition_prob = probability of reaching s1 given s0,a -> p(s1|s0, a)
				# action_prob = pi(a|s) -> probability of taking action a given state s and policy pi
				s1_val = self.V[s1_idx]  # value of s1

				if 0 < transition_prob < 1 and s1_val > 0:
					a =1 

				#action value: pi(a|s0)    * p(s1|s0,a)      *  [R(s)+V(s1)]
				action_value = action_prob * transition_prob *  (r + s1_val)
				value += action_value # the value is the sum across all action values

		self.V[idx] = value
		
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
			[valid_actions.append((state_idx, action, p)) for action, p in enumerate(state) if p > 0]

		return valid_actions

	def walk_with_state_transitions(self, start=None, probabilistic = False, avoid_states=[]):
		if start is None:
			curr = self.env.start.copy()
		else: 
			curr = self.second_start_position.copy()

		walk, walk_idxs = [], []

		reached_goal = False
		for step in np.arange(50):
			walk.append(curr)

			current_index = self.get_state_index(curr)
			walk_idxs.append(current_index)
			valid_actions = self.get_valid_actions(curr)
			
			if avoid_states: # avoid going to states visited during previous walks
				if step > 3:
					valid_actions = [(s,a,p) for s,a,p in valid_actions if s not in avoid_states and a not in [3] and s not in walk_idxs]
					if not valid_actions: break

			values = [self.V[si] for si,a,p in valid_actions]

			if not probabilistic: # choose the action lading to the state with the highest value
				selected = self.env.free_states[valid_actions[np.argmax(values)][0]]
			else:  # choose the actions with a probability proportional to their relative value
				selected = self.env.free_states[random.choices(valid_actions, weights=softmax(values), k=1)[0][0]]
			curr = selected

			if curr == self.env.goal or dist(self.env.goal, curr) < 2: 
				reached_goal = True
				break

		if reached_goal:
			walk.append(curr)
			return walk
		# else:
		# 	print("Walk didnt reach the shelter sorry")


	def do_probabilistic_walks(self, n=100):
		visited, walks = [], []
		for i in np.arange(n):
			walk = self.walk_with_state_transitions(probabilistic=True, avoid_states=visited)
			if walk is None: continue
			else:
				visited.extend([self.get_state_index(s) for s in walk])
				walks.append(walk)

		f, ax = plt.subplots()
		ax.imshow(self.env.maze_image, cmap="Greys_r")
		for walk in walks:
			if walk is not None:
				self.plot_walk(walk, ax=ax, background=False, multiple=True)


	def step(self, policy, current):
		legal_actions = [a for a in self.env.get_available_moves(current) if a != "still" not in a]
		action = choice(legal_actions)

		# move
		action_n = [k for k,v in self.actions.items() if v == action][0]
		return self.enact_step(current, action), action_n

	def introduce_blockage(self, bridge, p=.7):
		if 'lambda' in bridge: blocks = self.states_to_block_mb_lambda
		elif bridge=='alpha1': blocks = self.states_to_block_mb_alpha1
		elif bridge=='alpha0': blocks = self.states_to_block_mb_alpha0
		else:
			blocks = self.states_to_block_mb_alpha1
			blocks.extend(self.states_to_block_mb_alpha0)

		actions_to_block = [0, 1, 2, 4, 5] # Only actions between left, up, right should be blocked
		
		for state in blocks:
			try:
				idx = self.get_state_index(state)
			except:
				pass
			else:
				# Get all the states-actions leading to state and set the probability to p
				self.P[:, idx] = np.where(self.P[:, idx], p, self.P[:, idx])
				
			self.env.maze_image[state[1], state[0]] = p
			




