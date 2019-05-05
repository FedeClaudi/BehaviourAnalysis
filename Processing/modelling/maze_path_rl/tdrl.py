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

from tqdm import tqdm

class Agent:
	def __init__(self, env):
		self.env = env
		self.actions()

		self.second_start_position = [9, 9]

		# Save a copy of the initial free_states
		self.free_states = self.env.free_states.copy()
		self.maze_image = self.env.maze_image.copy()

		self.states_to_block_mb_lambda = [[9, 5], [10, 5], [9, 6], [10, 6]]
		self.states_to_block_mf_lambda = [x[::-1] for x in self.states_to_block_mb_lambda]

		self.states_to_block_mb_alpha1 = [[10, 9], [11,9]]
		self.states_to_block_mf_alpha1 = [x[::-1] for x in self.states_to_block_mb_alpha1]

		self.states_to_block_mb_alpha0 = [[8, 9], [9,9]]
		self.states_to_block_mf_alpha0 = [x[::-1] for x in self.states_to_block_mb_alpha0]

	def actions(self):
		self.actions = self.env.actions
		self.n_actions = len(self.actions.keys())-1
		self.actions_lookup = list(self.actions.values())

	def enact_step(self, current, action):
		nxt = current.copy()
		up, down = -1, 1
		left, right = -1, 1
		if action == "down":
			nxt[1] += down
		elif action == "right":
			nxt[0] += right
		elif action == "left":
			nxt[0] += left
		elif action == "up":
			nxt[1] += up
		elif action == "up-right":
			nxt[0] += right
			nxt[1] += up
		elif action == "up-left":
			nxt[0] += left
			nxt[1] += up
		elif action == "down-right":
			nxt[0] += right
			nxt[1] += down
		elif action == "down-left":
			nxt[0] += left
			nxt[1] += down
		return nxt

	def plot_walk(self, walk, ax=None, blocked=None, background=True, multiple=False):
		if ax is None:
			f,ax = plt.subplots()

		if background: 
			ax.imshow(self.env.maze_image, cmap="Greys_r")

		if not multiple:
			ax.scatter(np.vstack(walk)[:, 0], np.vstack(walk)[:, 1], c = np.arange(len(walk)), s=150)
		else:
			ax.scatter(np.vstack(walk)[:, 0], np.vstack(walk)[:, 1], alpha=.4, s=100)

		# if blocked:
		# 	ax.scatter([y for x,y in blocked], [x for x,y in blocked], c='r', s=250, alpha=.4)
		ax.set(xticks=[], yticks=[])

	def get_state_index(self, state):
		return [i for i,f in enumerate(self.env.free_states) if f == state][0]


class TDRL(Agent):
	def __init__(self, env):
		Agent.__init__(self, env)
		# Define empty Q table and learning params
		self.Q = self.empty_policy()

		# Parameters
		self.max_iters = 300
		self.max_steps = round(self.env.grid_size**2 / 2)

		self.epsilon = .95  # the higher the less greedy
		self.alpha = 1     # how much to value new experience, in a deterministic world set as 1
		self.gamma = .8    # discount on future rewards, the higher the less discount

		

	def empty_policy(self):
		return np.zeros((self.env.grid_size, self.env.grid_size, len(self.actions)))

	def run(self):
		# prep figure
		f, axarr =  plt.subplots(ncols=5, nrows=2)

		# Train on vanilla environment
		self.train(plot=False)

		# Plot the shortest path given the environment
		walk = self.shortest_walk_f()
		self.plot_policy(ax=axarr[0, 0], title="Model Free - Naive")
		self.plot_walk(walk, ax=axarr[1, 0])

		# Block LAMBDA and get the shortest path
		self.introduce_blockage('lambda')
		walk = self.shortest_walk_f()
		self.plot_policy(ax=axarr[0, 1], title="Bocked LAMBDA")
		self.plot_walk(walk, ax=axarr[1, 1], blocked = self.states_to_block_mf_lambda)

		# Train with blocked lambda to visualise policy update
		self.train(plot=False)

		# Re train (to re-initialise stuff)
		self.reset_trained()

		# Block ALPHA and get the shortest path
		self.introduce_blockage('alpha0')
		walk = self.shortest_walk_f()
		self.plot_policy(ax=axarr[0, 2], title="Bocked ALPHA0")
		self.plot_walk(walk, ax=axarr[1, 2], blocked = self.states_to_block_mf_alpha0)

		# Train with blocked alpha to visualise policy update
		self.train(plot=False)

		# Re train (to re-initialise stuff)
		self.reset_trained()

		# Block both ALPHA and get the shortest path
		self.introduce_blockage('alpha')
		walk = self.shortest_walk_f()
		self.plot_policy(ax=axarr[0, 3], title="Bocked ALPHAs")
		self.plot_walk(walk, ax=axarr[1, 3], blocked = self.states_to_block_mf_alpha0)

		# Reset again
		self.reset_trained()

		# Make a trial from secondary start point
		self.introduce_blockage('lambda')
		walk = self.shortest_walk_f(start='secondary')
		self.plot_policy(ax=axarr[0, 4], title="Start at P")
		self.plot_walk(walk, ax=axarr[1, 4],  blocked = self.states_to_block_mf_lambda)

		# reset but dont trian
		self.reset_trained(train=False)


	def reset_trained(self, train=True):
		self.env.free_states = self.free_states.copy()  # restore original free states
		self.env.maze_image = self.maze_image.copy()		
		if train: self.train()

	def train(self, plot=False):
		print("Training")
		not_change_count = 0

		if plot:
			f, axarr = plt.subplots(nrows=7, ncols=7)
			axarr = axarr.flatten()
			curr_ax = 0

		# iterate the learning
		for iter_n in tqdm(np.arange(self.max_iters)):

			# If plotting plot
			if plot and iter_n % 5 == 0 and curr_ax < len(axarr):
				self.plot_policy(axarr[curr_ax], title="iter - "+str(iter_n))
				self.plot_walk(self.shortest_walk_f(), ax=axarr[curr_ax])

				curr_ax += 1

			# reset starting position
			self.env.reset()

			# reset variables
			game_over = False
			step = 0
			walk = []
			# keep stepping and evaluating reward
			while not (game_over or step > self.max_steps):
				step += 1
				curr_state = self.env.state()
				walk.append(curr_state)

				if np.random.rand() <= self.epsilon:  # epsilon-greedy policy
					action = np.random.randint(0, self.n_actions)  # random action
				else:
					action = np.argmax(self.Q[curr_state[0]][curr_state[1]]) # best action from Q table

				# move in the environment
				next_state, reward, game_over = self.env.act(action)

				# Q-learning update
				curr_state_action = self.Q[curr_state[0], curr_state[1], action]
				next_state_best_action = max(self.Q[next_state[0], next_state[1], :])
				delta = self.gamma * next_state_best_action - curr_state_action
				self.Q[curr_state[0], curr_state[1], action] = curr_state_action + self.alpha*(reward + delta)

			# if plot:
			# 	self.plot_walk(walk)


	def step(self, current):
			# Select which action to perform
			action_values = self.Q[current[0], current[1], :]
			action = np.argmax(action_values)
			action = self.env.actions[action]
		
			# Move the agent accordingly
			return self.enact_step(current, action)


	def shortest_walk_f(self,  start=None, ttl=None):
		walk = []
		if start is None:
			curr = self.env.start.copy()
		else:
			curr = self.second_start_position.copy()

		step_n = 0
		# do each step
		while curr != self.env.goal and step_n < self.max_steps:
			step_n += 1
			walk.append(curr.copy())

			nxt = self.step( curr)

			# Check that nxt is a legal move
			if nxt[0] < 0 or nxt[1] < 0 or nxt[0] > self.env.grid_size or nxt[1] > self.env.grid_size:
				break
			if nxt in self.env.free_states:
				curr = nxt
		walk.append(curr)
		return walk

	def plot_policy(self, ax, title):
		policy = np.rot90(np.mean(self.Q, 2), 3)[:, ::-1]
		policy[policy == 0] = np.nan
		ax.imshow(policy)

		ax.set(title=title, xticks=[], yticks=[])

	def introduce_blockage(self, bridge):
		if 'lambda' in bridge: blocks = self.states_to_block_mf_lambda
		elif bridge=='alpha1': blocks = self.states_to_block_mf_alpha1
		elif bridge=='alpha0': blocks = self.states_to_block_mf_alpha0
		else:
			blocks = self.states_to_block_mf_alpha1
			blocks.extend(self.states_to_block_mf_alpha0)

		for block in blocks:
			self.Q[block[1], block[0], :] = 0

			self.env.maze_image[block[1], block[0]] = 0 
			self.env.maze_image[block[0], block[1]] = 0 

		self.env.free_states = [fs for fs in self.env.free_states if fs[::-1] not in blocks]



