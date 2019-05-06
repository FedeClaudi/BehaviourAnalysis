import sys
sys.path.append('./')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import random
from scipy.special import softmax

from Processing.modelling.maze_solvers.agent import Agent

class VanillaMF(Agent):
	def __init__(self):
		Agent.__init__(self)

		# Parameters
		self.max_iters = 300
		

		self.epsilon = .95  # the higher the less greedy
		self.alpha = 1     # how much to value new experience, in a deterministic world set as 1
		self.gamma = .8    # discount on future rewards, the higher the less discount

		self.probabilistic_walk = False # select each action based on its relative value and not just the one with max val

		# Define empty Q table and learning params
		self.Q = self.empty_policy()



	def empty_policy(self):
		return np.zeros((self.grid_size, self.grid_size, len(self.actions)))

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
		self.plot_walk(walk, ax=axarr[1, 1], blocked = self.states_to_block_lambda)

		# Train with blocked lambda to visualise policy update
		self.train(plot=False)

		# Re train (to re-initialise stuff)
		self.reset_trained()

		# Block ALPHA and get the shortest path
		self.introduce_blockage('alpha0')
		walk = self.shortest_walk_f()
		self.plot_policy(ax=axarr[0, 2], title="Bocked ALPHA0")
		self.plot_walk(walk, ax=axarr[1, 2], blocked = self.states_to_block_alpha0)

		# Train with blocked alpha to visualise policy update
		self.train(plot=False)

		# Re train (to re-initialise stuff)
		self.reset_trained()

		# Block both ALPHA and get the shortest path
		self.introduce_blockage('alpha')
		walk = self.shortest_walk_f()
		self.plot_policy(ax=axarr[0, 3], title="Bocked ALPHAs")
		self.plot_walk(walk, ax=axarr[1, 3], blocked = self.states_to_block_alpha0)

		# Reset again
		self.reset_trained()

		# Make a trial from secondary start point
		self.introduce_blockage('lambda')
		walk = self.shortest_walk_f(start='secondary')
		self.plot_policy(ax=axarr[0, 4], title="Start at P")
		self.plot_walk(walk, ax=axarr[1, 4],  blocked = self.states_to_block_lambda)

		# reset but dont trian
		self.reset_trained(train=False)


	def reset_trained(self, train=True):
		self.free_states = self._free_states.copy()  # restore original free states
		self.maze = self._maze.copy()
		if train: self.train()

	def train(self, plot=False):
		print("Training")

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
			self.reset()

			# reset variables
			self.game_over = False
			step = 0
			walk = []
			# keep stepping and evaluating reward
			while not (self.game_over or step > self.max_steps):
				step += 1
				curr_state = self.state()
	
				walk.append(curr_state)
				
				if np.random.rand() <= self.epsilon:  # epsilon-greedy policy
					action = np.random.randint(0, self.n_actions)  # random action
				else:
					action = np.argmax(self.Q[curr_state[0]][curr_state[1]]) # best action from Q table

				# move in the environment
				next_state, reward, game_over = self.act(action)

				# Q-learning update
				curr_state_action = self.Q[curr_state[0], curr_state[1], action]
				next_state_best_action = max(self.Q[self.next_state[0], self.next_state[1], :])
				delta = self.gamma * next_state_best_action - curr_state_action
				self.Q[curr_state[0], curr_state[1], action] = curr_state_action + self.alpha*(self.reward + delta)

			# if plot:
			# 	self.plot_walk(walk)


	def step(self, current):
			# Select which action to perform
			action_values = self.Q[current[0], current[1], :]

			
			if not self.probabilistic_walk:
				action = np.argmax(action_values)
			else:
				action = np.where(action_values == random.choices(action_values, weights=softmax(action_values), k=1)[0])[0][0]
		
			action = self.actions[action]

			# Move the agent accordingly
			return self.move(action, current)


	def shortest_walk_f(self,  start=None, ttl=None):
		walk = []
		if start is None:
			curr = self.start_location.copy()
		else:
			curr = self.second_start_location.copy()

		step_n = 0
		# do each step
		while curr != self.goal_location and step_n < self.max_steps:
			step_n += 1
			walk.append(curr.copy())

			nxt = self.step(curr)

			# Check that nxt is a legal move
			if nxt[0] < 0 or nxt[1] < 0 or nxt[0] > self.grid_size or nxt[1] > self.grid_size:
				break
			if nxt in self.free_states:
				curr = nxt
		walk.append(curr)
		return walk

	def plot_policy(self, ax, title):
		policy = np.rot90(np.mean(self.Q, 2), 3)[:, ::-1]
		policy[policy == 0] = np.nan
		ax.imshow(policy)

		ax.set(title=title, xticks=[], yticks=[])

	def introduce_blockage(self, bridge):
		blocks = self.get_blocked_states(bridge)

		for block in blocks:
			self.geodesic_distance[block[0], block[1], :] = 0

			self.maze[block[1], block[0]] = 0 
			self.maze[block[0], block[1]] = 0 

		self.free_states = [fs for fs in self.free_states if fs[::-1] not in blocks]


if __name__ == "__main__":
	agent = VanillaMF()
	agent.run()
	plt.show()






