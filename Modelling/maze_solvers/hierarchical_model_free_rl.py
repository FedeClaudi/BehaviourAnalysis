import sys
sys.path.append('./')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import random
from scipy.special import softmax


from Processing.modelling.maze_solvers.agent import Agent

class HMF(Agent):
	def __init__(self):
		Agent.__init__(self)

		# Parameters
		self.max_iters = 50
		

		self.epsilon = .95  # the higher the less greedy
		self.alpha = 1     # how much to value new experience, in a deterministic world set as 1
		self.gamma = .8    # discount on future rewards, the higher the less discount

		self.goal_reward = 0
		self.step_punishment = 0.5

		self.probabilistic_walk = False # select each action based on its relative value and not just the one with max val

		# Define empty Q table and learning params
		self.Q = self.empty_policy()

	def empty_policy(self):
		return {k:0 for k in list(self.options.keys())}


	def train_on_options(self):
		"""[Select each option either based on its value or at random and compute the value of that option]
		"""
		for iter_n in tqdm(np.arange(self.max_iters)):
			if np.random.rand() <= self.epsilon:  # epsilon-greedy policy
				selected_option = random.choice(list(self.options.keys()))
			else:
				selected_option = random.choices(list(self.Q.keys()), weights=softmax(list(self.Q.values())), k=1)[0]
			self.enact_option(selected_option)

		print("Learned options: ", self.Q)
		print("Relative likelihood: ", [round(x,2) for x in softmax(list(self.Q.values()))] )


	def enact_option(self, selected_option):
		"""[Each option receives a reward equal to reaching the goal and a punishment equal to the number of steps necessary to reach the goal,
				these quantities are used to update the value of the option]
		
		Arguments:
			selected_option {[string]} -- [option name]
		"""
		# compute reward
		reward = self.goal_reward  - (self.step_punishment * len(self.options[selected_option]))

		# update value
		rpe = reward - self.Q[selected_option]
		self.Q[selected_option] = self.Q[selected_option] + (self.alpha * rpe)



	def train_on_actions(self):
		# TODO: this train a standard model free agent and than use that agents Q table to compute the alues of each option for the hierarchical agent
		pass


	def simulate_probabilistic_choice(self):
		f, ax = plt.subplots()

		outcomes = [random.choices(list(self.Q.keys()), weights=softmax(list(self.Q.values())), k=1)[0] for x in np.arange(1000)]

		ax.hist(outcomes)


if __name__ == "__main__":
	hmf = HMF()
	hmf.train_on_options()

	hmf.show()