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



class Agent:
	def __init__(self, env):
		self.env = env
		self.actions()

		self.max_iters = 51
		self.max_steps = round(self.env.grid_size**2 / 2)

	
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

	def plot_walk(self, walk, title=None):
		f, ax = plt.subplots()

		ax.imshow(self.env.maze_image, cmap="Greys_r")
		ax.scatter(np.vstack(walk)[:, 0], np.vstack(walk)[:, 1],
				c = np.arange(len(walk)), s=50)
		ax.set(title=title)

	def get_state_index(self, state):
		return [i for i,f in enumerate(self.env.free_states) if f == state][0]


class TDRL(Agent):
	def __init__(self, env):
		Agent.__init__(self, env)
		# Define empty Q table and learning params
		self.policies()

		# Parameters
		self.epsilon = .95  # the higher the less greedy
		self.alpha = 1      # how much to value new experience, in a deterministic world set as 1
		self.gamma = .9     # discount on future rewards, the higher the less discount


	def empty_policy(self):
		#temp =  [[list(np.zeros(len(self.actions.keys()))) for i in range(self.env.grid_size)] for j in range(self.env.grid_size)]
		
		return np.zeros((self.env.grid_size, self.env.grid_size, len(self.actions)))

	def policies(self):
		# Get basic policies
		self.Q = self.empty_policy()

	def train(self):
		print("Training")
		not_change_count = 0

		# iterate the learning
		for iter_n in np.arange(self.max_iters):
			if iter_n % 100 == 0: print("iteration: ", iter_n)

			# reset starting position
			self.env.reset()

			# reset variables
			game_over = False
			step = 0
			Q2 = self.Q.copy()

			# keep stepping and evaluating reward
			while not (game_over or step > self.max_steps):
				step += 1
				curr_state = self.env.state()

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


	def step(self, current):
			"""[steps the agent while walking the maze. Given a certain policy and state (current position), selects the best 
			actions and moves the agent accordingly]
			
			Arguments:
				policy {[str]} -- [name of the self.Q entry - i.e. policy to sue while evaluating which action to take]
				current {[list]} -- [x,y coordinates of current position]
				random_action {[bool]} [if we should take a random action instead of using the policies to select an acyion]
			
			Returns:
				[type] -- [description]
			"""

			# Select which action to perform
			action_values = self.Q[current[0], current[1], :]
			action = np.argmax(action_values)
			action = self.env.actions[action]
		
			# Move the agent accordingly
			return self.enact_step(current, action)


	def shortest_walk_f(self,  start=None):
		"""[Using the sheler policy, it finds the shortest path to the shelter]
		

		Keyword Arguments:
			start {[list]} -- [alternative start to default one, optional] (default: {None})
		
		Returns:
			[list] -- [list of x,y coordinates at each step n the walk]
		"""

		walk = []
		if start is None:
			curr = self.env.start.copy()
		else:
			curr = start.copy()

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

		self.shortest_walk = walk

		self.plot_walk(walk)
		return walk




