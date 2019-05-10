import sys
sys.path.append('./')

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from random import choice
import pandas as pd
import random
import pickle
from scipy.special import softmax
from tqdm import tqdm 
from mpl_toolkits.mplot3d import Axes3D

from Processing.tracking_stats.math_utils import geodist

from Processing.modelling.maze_solvers.agent import Agent


class StraightRunner(Agent):
	def __init__(self):
		Agent.__init__(self)

		self.current_direction = None
		self.past_state = None

	def run(self):
		for i in range(10):
			self.plot_walk(self.walk())

	def step(self):
		return self.move(self.current_direction, self.curr_state)


	def walk(self, start=None):
		"""[this agent runs by selecting a random direction among the ones available and running straight until an obstacle is encountered]
		"""

		walk = []
		if start is None:
			curr = self.start_location.copy()
		else:
			curr = self.second_start_location.copy()
		walk.append(curr)

		step_n = 0

		# start by selecting a random direction
		self.current_direction = self.select_random_running_direction()

		# do each step
		while curr != self.goal_location and step_n < self.max_steps:
			step_n += 1

			nxt = self.step()

			while nxt not in self.free_states:
				self.select_random_running_direction()
				nxt = self.step()
				curr = nxt
				self.curr_state = curr
				walk.append(curr)
		
			curr = nxt
			self.curr_state = curr
			walk.append(curr)
		return walk
			
	def select_random_running_direction(self):
		self.current_direction = random.choice(self.get_available_moves())
		# print("at: ", self.curr_state,  " running towards: ", direction)



if __name__ == "__main__":
	sr = StraightRunner()
	sr.run()
	plt.show()
