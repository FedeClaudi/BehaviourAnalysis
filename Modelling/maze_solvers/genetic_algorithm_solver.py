import sys
sys.path.append('./')
import matplotlib.pyplot as plt
import numpy as np

import os
from random import choice, choices, sample, shuffle
import pandas as pd
from tqdm import tqdm 
import time


from Processing.modelling.maze_solvers.agent import Agent
from Processing.tracking_stats.math_utils import calc_distance_from_shelter as goal_dist


class GA(Agent):
	def __init__(self):
		Agent.__init__(self)

		self.pop_size = 250
		self.max_n_steps = 100
		self.n_generations = 50
		self.keep_best = 10
		self.mutation_rate = .1 # probability of a gene mutating

		self.best_worst_history = []
		self.best_walks_history = []

		self.we_got_a_winner = False

		self.initialise_genomes()
		self.evolve()



	def initialise_genomes(self):
		"""[The each gene in the genome corresponds to a free state in the maze, the value of each gene corresponds to
			the action being taken. The genome is initialised with random genes]
		"""
		self.free_states_lookup = {str(i):state for i,state in enumerate(self.free_states)}
		# self.genomes = {str(i):{k:np.random.randint(0, self.n_actions+1, 1)[0] for k in self.free_states_lookup.keys()} for i in np.arange(self.pop_size)}
		
		self.genomes = {"individual - {}".format(i):np.random.randint(0, self.n_actions, len(self.free_states)) for i in np.arange(self.pop_size)}
		self.individuals = pd.DataFrame(self.genomes.items(), columns=["name", "genome"])
		self.individuals['fitness'] = ""
		self.individuals['walk'] = ""
		self.individuals['walk_dist'] = ""

	def visualise_genome(self):
		plt.figure()
		genomes_array = np.vstack([list(list(self.genomes.values())[i].values()) for i in np.arange(self.pop_size)])
		plt.imshow(genomes_array)

	"""
	======================================================================================================
	======================================================================================================
	======================================================================================================
	"""

	def evolve(self):
		for iter_n in np.arange(self.n_generations):
			start_time = time.time()
			print("\n\nRunning generation {} of {}".format(iter_n+1, self.n_generations))
			self.run_generation()

			# fitness is the sum of tot reward plus tot dist walked
			self.individuals['tot_fitness'] = np.int32(np.subtract(self.individuals.fitness.values, np.divide(self.individuals.walk_dist.values, 10)))
			self.best_walks_history.append(self.individuals.sort_values("tot_fitness").loc[0].walk)

			print("		worst: {}\n		best: {}".format(np.nanmin(self.individuals.tot_fitness), np.nanmax(self.individuals.tot_fitness)))

			# self.plot_fitness()
			# self.plot_best_vs_worst()

			self.best_worst_history.append((np.min(self.individuals.tot_fitness), np.max(self.individuals.tot_fitness)))
			if self.we_got_a_winner: break
			if iter_n < self.n_generations-1: self.make_new_gen()

			end_time = time.time()
			duration = round(end_time - start_time)
			left_iters = round(self.n_generations - iter_n - 1)
			print("		iter duration: {}s, {} min for remaining {} iters".format(duration, round((duration*left_iters)/60), left_iters))


		self.visualise_end_state()

	def visualise_end_state(self):
		f, axarr = plt.subplots(ncols=4)

		walks = [self.best_walks_history[0], self.best_walks_history[3], self.best_walks_history[5], self.best_walks_history[-1]]

		# walks = shuffle(self.best_walks_history) [5]

		for ax, walk in zip(axarr, walks):
			self.plot_walk(walk, ax=ax)



	def run_generation(self):
		for i_id, individual in tqdm(self.individuals.iterrows()):
			self.reset()
			walk, com_rew = [], 0

			curr_state = self.state()
			walk.append(curr_state)

			for step_n in np.arange(self.max_n_steps):
				if step_n > 0: curr_state = next_state
				curr_idx = self.get_state_index(curr_state)

				action = individual.genome[curr_idx]
				next_state, reward, game_over = self.act(action, mode="genetic")
				com_rew += reward
				walk.append(next_state)


				if game_over: 
					self.we_got_a_winner = True
					break

			#individual["comulative reward"] = com_rew
			
			self.individuals.at[i_id, "fitness"] = com_rew
			self.individuals.at[i_id, "walk"] = walk
			self.individuals.at[i_id, "walk_dist"] = np.sum(goal_dist(np.vstack(walk), self.goal_location))

	def make_new_gen(self):
		# discard low fitness individuals
		self.individuals = self.individuals.sort_values("tot_fitness", ascending=False)
		self.individuals = self.individuals[:self.keep_best]

		#  replenish population
		n_couples = (self.pop_size - len(self.individuals)) / 2
		children = pd.DataFrame(columns = self.individuals.columns)
		for couple in np.arange(n_couples):
			# select two random parents
			parents = self.individuals.sample(n=2, replace=False)
			parents = parents.genome.values

			# make names for the 2 children
			c1 = ["individual - {}".format(max(self.individuals.index) + len(children) + 1)]
			c2 = ["individual - {}".format(max(self.individuals.index) + len(children) + 1)]

			# recombine genomes
			split_point = np.random.randint(0, len(parents[0]), 1)[0]

			c1.extend([np.hstack([parents[0][:split_point], parents[1][split_point:]]), "", "", "", ""])
			c2.extend([np.hstack([parents[1][:split_point], parents[0][split_point:]]), "", "", "", ""])

			# mutate genes
			n_mutated_genes = np.int(len(c1[1])*self.mutation_rate)

			mutated_idx = np.random.choice(np.arange(len(c1[1])), size=n_mutated_genes, replace=False)
			c1[1][mutated_idx] = np.random.randint(0, self.n_actions, n_mutated_genes)
			mutated_idx = np.random.choice(np.arange(len(c1[1])), size=n_mutated_genes, replace=False)
			c2[1][mutated_idx] = np.random.randint(0, self.n_actions, n_mutated_genes)


			# add to childresn df
			children.loc[-1] = c1
			children.loc[-2] = c2
			children.index = children.index + 2  # shifting index
			children = children.sort_index() 

		# add children to survived parents
		self.individuals = self.individuals.append(children, ignore_index=True)
		self.individuals = self.individuals.sort_index()


		"""
		======================================================================================================
		======================================================================================================
		======================================================================================================
		"""

	def plot_best_vs_worst(self):
		walks = [self.individuals.sort_values("fitness").loc[0].walk, self.individuals.sort_values("fitness").loc[len(self.individuals)-1].walk]
		self.plot_walks(walks)

	def plot_best_vs_worst_history(self):
		f, ax = plt.subplots()
		ax.plot([s[0] for s in self.best_worst_history], color='r', label="worst")
		ax.plot([s[1] for s in self.best_worst_history], color='k', label="best")
		ax.legend()

	def plot_fitness(self):
		plt.figure()
		plt.plot(sorted(self.individuals.fitness.values))



if __name__ == "__main__":
	ga = GA()
	ga.evolve()

	plt.show()
