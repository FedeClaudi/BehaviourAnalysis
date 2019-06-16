# %%
import sys
sys.path.append("./")
from Utilities.imports import *
from sklearn.utils.extmath import softmax
from random import choice, choices
%matplotlib inline  


from Modelling.maze_solvers.agent import Agent
from Utilities.maths.math_utils import calc_distance_from_shelter as goal_dist


# TODO chrome-extension://bomfdkbfpdhijjbeoicnfhjbdhncfhig/view.html?mp=bGVfA9sg
# %%
class GA(Agent):
	def __init__(self, *args, **kwargs):
		Agent.__init__(self, goal_location=[20, 12], start_location=[20, 28])
		# ? params
		self.pop_size = 50
		self.max_n_steps = 50
		self.n_generations = 50
		self.keep_best = 5
		self.mutation_rate = .05 # probability of a gene mutating

	def get_maze_designs(self):
		images = [f for f in os.listdir(self.maze_models_folder) if "png" in f]
		images = [self.get_maze_from_image(os.path.join(self.maze_models_folder, i)) for i in images]
		
		self.images = images
		return images

	def plot_all_mazes(self):
		for i,fs in self.images:
			f,ax = plt.subplots()
			ax.imshow(i)
			ax.plot(self.start_location[0], self.start_location[1], 'o', color='r')
			ax.plot(self.goal_location[0], self.goal_location[1], 'o', color='g')

	def initialise_genomes(self):
		"""
			Each genome is an Nxa (N = # free states, a = # actions) matrix with each "gene" mapping onto a state in the maze image space
			Each gene has m numbers to it, each of them representing the value of each action in that state. 
			The actions will then be selected with softmax
			The genomes are initialised with random value
		"""
		self.genomes = np.array([np.random.randint(0, 8, size=(self.grid_size*self.grid_size)) for i in np.arange(self.pop_size)])

	def evolve_open_arena(self):
		# crate an open arena maze
		self.maze = np.ones((self.grid_size, self.grid_size))
		self.free_states = [[i,j] for i in np.arange(self.grid_size) for j in np.arange(self.grid_size)]
		
		# do generations
		fitness_record = []
		for iter_n in np.arange(self.n_generations):
			# let each individual do it's thing
			# print("running generation - {}".format(iter_n))
			walks, walks_l, rewards, shelter_distance = self.generation_step()
			fitness = np.array([r+d-l for l,r,d in zip(walks_l, rewards, shelter_distance)])
			fitness = fitness.flatten()
		
			# sort by fitness and keep the best
			sort_idx = np.argsort(fitness)
			self.genomes = self.genomes[sort_idx][:self.keep_best]

			fitness = fitness[sort_idx][:self.keep_best]
			fitness_record.append(np.max(fitness))
			walks = [walks[i] for i in sort_idx]

			# Re populate
			# print("re populating")
			if iter_n < self.n_generations:
				self.make_new_gen()

		# finished
		print("finished, final fitness:")
		self.plot_fitness_history(fitness_record, walks[0])

	def generation_step(self):
		walks, walks_l, rewards, shelter_distance = [], [], [], []
		for genome in tqdm(list(self.genomes)):
			self.reset()
			
			# initialise walk
			curr_state = self.state()
			walk = [curr_state]
			com_rew = 0

			# walk
			for step_n in np.arange(self.max_n_steps):
				if step_n > 0: curr_state = next_state
				curr_idx = self.get_state_index(curr_state)

				# select and action 
				# action_vals =  softmax(genome[curr_idx, :].reshape(-1, 1).T)
				# action = choices(list(self.actions.keys()), weights=action_vals.T, k=1)[0]
				action = list(self.actions.keys())[(genome[curr_idx])]

				# enact
				next_state, reward, game_over = self.act(action, mode="genetic")
				com_rew += reward
				walk.append(next_state)

				# evaluate
				if game_over: 
					self.we_got_a_winner = True
					break

			# keep the results
			walks.append(walk)
			walks_l.append(len(walk))
			rewards.append(com_rew)
			shelter_distance.append(goal_dist(np.array(walk[-1]).reshape(1,2), self.goal_location))
		return walks, walks_l, rewards, shelter_distance

	def make_new_gen(self):
		parents_gen = self.genomes.copy()

		while len(self.genomes) < self.pop_size:
			# take two random parents
			parents = [np.array(g) for g in choices(parents_gen, k=2)]

			# recombine them genomes
			split_point = np.random.randint(0, len(parents[0]), 1)[0]
			c1 = np.hstack([parents[0][:split_point], parents[1][split_point:]])
			c2 = np.hstack([parents[1][:split_point], parents[0][split_point:]])

			# mutate a few genes
			n_mutated_genes = np.int(len(c1)*self.mutation_rate)
			c1_mutate = choices(np.arange(len(c1)), k=n_mutated_genes)
			c2_mutate = choices(np.arange(len(c1)), k=n_mutated_genes)
			# c1[c1_mutate] = np.random.uniform(0, 1, size=(n_mutated_genes, 8))
			c1[c1_mutate] = np.random.randint(0, 8, size=n_mutated_genes)
			c2[c2_mutate] = np.random.randint(0, 8, size=n_mutated_genes)

			print(self.genomes.shape, c1.shape)
			np.append(self.genomes, c1)
			np.append(self.genomes, c2)
			print(self.genomes.shape, c1.shape)
			break


	def plot_fitness_history(self, fitness_record, walk):
		f, axarr = plt.subplots(nrows=2)
		axarr[0].plot(fitness_record, color="r")
		self.plot_walk(walk, ax=axarr[1])

# %%
ga = GA()
# imgs = ga.get_maze_designs()

#%%
ga.initialise_genomes()
ga.evolve_open_arena()

#%%


#%%
