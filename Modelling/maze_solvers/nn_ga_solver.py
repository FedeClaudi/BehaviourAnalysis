# %%
import sys
sys.path.append("./")
from Utilities.imports import *
from sklearn.utils.extmath import softmax
from random import choice, choices
# %matplotlib inline  


from Modelling.maze_solvers.agent import Agent
from Utilities.maths.math_utils import calc_distance_from_shelter as goal_dist

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools

# %%
# individual class
class AgentNet:
	# see: https://towardsdatascience.com/coding-a-2-layer-neural-network-from-scratch-in-python-4dd022d19fd2
	def __init__(self, genome):
		self.dims = [8, 8, 1]
		self.prams = {}

		# weights and biases are specified in the genome on separate chromosomes
		self.weights = [genome[0].reshape(8, 8), genome[1].reshape(8, 8)]
		self.bias = [genome[2], genome[3]]


	@staticmethod
	def Sigmoid(Z):
		return 1/(1+np.exp(-Z))

	@staticmethod
	def Relu(Z):
		return np.maximum(0,Z)

	def forward(self, X):    
		# run forward calculations
		# first layer
		Z1 = self.weights[0].dot(X) + self.bias[0]
		A1 = self.Relu(Z1)
		
		# second layer
		Z2 = self.weights[1].dot(A1) + self.bias[1]  
		A2 = self.Sigmoid(Z2)

		return A2, A1

# %% 
# test net
n_genes = [64, 64, 8, 8] 
data = np.array([1, 1, 1, 0, 0, 0, 0, 0])
genome = [np.random.uniform(0.0, 1.0, size=n) for n in n_genes]
nn = AgentNet(genome)
a1, a2 = nn.forward(data)

# %%
# Agent class
class NNGa(Agent):
	def __init__(self):
		grid_size = 40
		x = int(grid_size/2)-2
		ys, yt = int(grid_size/4)+2, int(grid_size/2)+int(grid_size/5)
		Agent.__init__(self, grid_size=grid_size, goal_location=[x, ys], start_location=[x, yt])

		# ? frequently used params
		self.multiple_mazes = False
		self.randomise_start_location = True
		self.deterministic = True


		# ? params
		self.pop_size = 50
		self.max_n_steps = 100 # grid_size*10
		self.n_generations = 300
		self.keep_best_n = 10
		self.mutation_rates = [.4, .25, .01, .01] # probability of a gene mutating
		self.mutation_rate = self.mutation_rates[0]
		self.learning_steps = [0, 20, 45, 70]

		# ? Deterministic actions
		if self.deterministic:
			self.repeat_gen = 1
		else:
			self.repeat_gen = 4

		# ? maze design
		self.get_maze_designs()
		img = self.images["mazemodel"]
		self.maze, self.free_states = self.get_maze_from_image(img)
		self.weights_max = 1.0 # max val for NN weights

		# ? Keep the best agent
		self.best_fitness = 0
		self.best_agent, self.best_genome = None, []

		if sys.platform == "darwin":
			self.save_fld = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/analysis_metadata/maze_solvers"
		else:
			self.save_fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\maze_solvers"

	def get_maze_designs(self):
		self.images = {f.split(".")[0]: os.path.join(self.maze_models_folder, f) for f in os.listdir(self.maze_models_folder) if "png" in f}
		# self.images = {i.split(".")[0] : self.get_maze_from_image(os.path.join(self.maze_models_folder, i)) 
		# 			for i in images}

	def initialise_genomes(self):
		# Genomes consist of twon chromosomes 64+8 genes, the weights and bias of each agent 
		n_genes = [64, 64, 8, 8] 
		self.genomes = [[np.random.uniform(0.0, self.weights_max, size=n) for n in n_genes] for i in range(self.pop_size)]

	def initialise_agents(self):
		self.agents = [AgentNet(genome) for genome in self.genomes]

	def evolve_open_arena(self):
		# crate an open arena maze
		self.maze = np.ones((self.grid_size, self.grid_size))
		self.free_states = [[i,j] for i in np.arange(self.grid_size) for j in np.arange(self.grid_size)]

		self.evolve(reward_modifier=+1)

	def evolve_maze(self):
		# self.plot_maze()
		self.evolve()

	def evolve(self, reward_modifier=0):
		# keep a record of the best of each gen
		self.best_walks, best_rewards, max_steps, min_dist = [], [], [], []

		# do generations
		fitness_record = []
		for iter_n in tqdm(np.arange(self.n_generations)):
			# checkif we need to change learning rate
			if iter_n in self.learning_steps:
				self.mutation_rate = self.mutation_rates[self.learning_steps.index(iter_n)]

			# letagents do their things
			repeat_rewards, repeat_distance, repeat_walks_steps = [], [], []
			for i in range(self.repeat_gen): # run each gen multiple times because of probabilistic agent
				# Choose a random maze
				if self.multiple_mazes:
					img = random.choice(list(self.images.values()))
					self.maze, self.free_states = self.get_maze_from_image(img)

				
				walks, rewards, shelter_distance, walks_n_steps = self.run_generation(reward_modifier=reward_modifier)
				repeat_rewards.append(rewards)
				repeat_distance.append(shelter_distance)
				repeat_walks_steps.append(walks_n_steps)
			
			rewards = np.mean(np.vstack(repeat_rewards), 0)  # make sure all arrays have correct size
			shelter_distance = np.mean(np.hstack(repeat_distance), 1)
			walks_n_steps = np.mean(np.vstack(repeat_walks_steps), 0)

			fitness = np.array([r-d**2-l for r,d,l in zip(rewards, shelter_distance, walks_n_steps)])  # ! fitness
			# fitness = np.array([r-d-l**2 for r,d,l in zip(rewards, shelter_distance, walks_n_steps)])  # ! fitness

			fitness = fitness.ravel()
			# fitness = rewards.copy()

			# sort based on performance and keep N best
			sort_idx = np.argsort(fitness)[::-1]

			fitness = np.array(fitness)[sort_idx]
			self.agents = [self.agents[i] for i in sort_idx][:self.keep_best_n]
			self.genomes = [self.genomes[i] for i in sort_idx][:self.keep_best_n]
			walks = [walks[i] for i in sort_idx]
			max_steps.append([walks_n_steps[i] for i in sort_idx][0])

			# keep data for plotting
			self.best_walks.append(walks[0])
			best_rewards.append(fitness[0])
			min_dist.append(shelter_distance[-1]*10)

			# replenish generation
			self.fill_generation()

			# keep a copy of the fittest agent
			if fitness[0] > self.best_fitness:
				self.best_fitness = fitness[0]
				self.best_agent = self.agents[0]
				self.best_genome.append(self.genomes[0])

			# ? print
			print(iter_n, "- f:", round(fitness[0], 2), "- lr:", self.mutation_rate, " - n steps: ", len(walks[0]))
			if fitness[0] != np.max(fitness): raise ValueError

		# save the best genome and plot
		# save_yaml(os.path.join(self.save_fld, "best_genome2.yml"), self.best_genome[-1])
		genome = pd.DataFrame(self.genomes[-1], index=["w1", "w2", "b1", "b2"]).T
		genome.to_pickle(os.path.join(self.save_fld, "best_genome.pkl"))
		self.plot_rewards_history(best_rewards, max_steps, min_dist)

	def run_generation(self, reward_modifier = 0):
		walks, walks_n_steps, rewards, shelter_distance = [], [], [], []
		for agent in self.agents:
			self.reset()

			# walk
			walk, com_rew = self.walk_agent(agent, reward_modifier, stop_wall=True, stop=True) # ! stop wall!!

			# extract metrics
			walks_n_steps.append(len(walk))
			shelter_distance.append(goal_dist(np.array(walk[-1]).reshape(1,2), self.goal_location))
			walks.append(walk)
			rewards.append(com_rew)
		return walks, rewards, shelter_distance, walks_n_steps

	def walk_agent(self, agent, reward_modifier, stop=True, stop_wall=False):
		keep_states = [x for x in np.arange(9) if x != 4]
		# initialise walk
		if self.randomise_start_location:
			self.curr_state = random.choice(self.free_states)

		curr_state = self.state()
		walk = [curr_state]
		com_rew = 0

		for step_n in np.arange(self.max_n_steps):
			curr_idx = self.get_state_index(curr_state)

			# Get the states around the agent
			_, surroundings = self.get_available_moves(current=curr_state)
			if surroundings is None: break
			surroundings = surroundings.flatten()[keep_states] # flatten and remove central state

			# use the nn controller to select an action in a probabilistic manner
			_, action_values = agent.forward(surroundings)
			
			if not self.deterministic:
				try:
					action = choices(list(self.actions.keys()), weights=action_values.T, k=1)[0]
				except: break

			else:
				action = list(self.actions.keys())[np.argmax(action_values)]

			# enact
			next_state, reward, game_over = self.act(action, mode="genetic") # gets a 200 reward for reaching the shelter
			reward += reward_modifier 
			curr_state = next_state
			com_rew += reward
			walk.append(next_state)

			# evaluate
			if stop:
				if game_over: 
					self.we_got_a_winner = True
					break
				elif reward > 100:
					break

			if stop_wall and reward < 0: 
				a = 1
				break

		return walk, com_rew

	def fill_generation(self):
		parents_gen = self.genomes.copy()

		while len(self.genomes) < self.pop_size:
			# take two random parents
			parents = [np.array(g) for g in choices(parents_gen, k=2)]

			# recombine them genomes
			children_genomes = [[], []]
			for chrom_1, chrom_2 in zip(*parents):
				split_point = np.random.randint(0, len(chrom_1), 1)[0]
				c1 = np.hstack([chrom_1[:split_point], chrom_2[split_point:]])
				c2 = np.hstack([chrom_2[:split_point], chrom_1[split_point:]])

				# mutate a few genes
				n_mutated_genes = np.int(len(c1)*self.mutation_rate)
				c1_mutate = choices(np.arange(len(c1)), k=n_mutated_genes)
				c2_mutate = choices(np.arange(len(c1)), k=n_mutated_genes)
				# c1[c1_mutate] = np.random.uniform(0, 1, size=(n_mutated_genes, 8))
				c1[c1_mutate] = np.random.randint(0, self.weights_max, size=n_mutated_genes)
				c2[c2_mutate] = np.random.randint(0, self.weights_max, size=n_mutated_genes)

				children_genomes[0].append(c1)
				children_genomes[1].append(c2)

			self.genomes.extend(children_genomes)

		self.initialise_agents()

	def plot_rewards_history(self, r_history, n_steps_history, min_dist):
		f, ax = plt.subplots()
		ax.plot(r_history, color="r", label="reward")
		ax.plot(n_steps_history, color="b", label="n steps")
		ax.plot(min_dist, color="g", label="min shelt d")
		ax.legend()

		f, ax = plt.subplots()
		for walk in self.best_walks[-3:]:
			self.plot_walk(walk, ax=ax)


# %%
# Analyse
class NNGaAnalyser(NNGa):
	def __init__(self):
		NNGa.__init__(self)
		genome = pd.read_pickle(os.path.join(self.save_fld, "good_genome.pkl"))

		self.genomes = [[genome.values[:n, i] for i, n in zip(range(4), [64, 64, 8, 8])]]
 
		self.initialise_agents()

	def plot_all_agents_walks(self):
		for agent in self.agents:
			self.reset()
			walk, _ = self.walk_agent(agent, 0,)
			self.plot_walk(walk)

	def plot_agent_walk(self, agent_n, restore_maze=False):
		f, ax = plt.subplots(figsize=(8, 8))

		walk_lengths = []
		for n in range(self.n_walks):
			self.reset()
			
			if restore_maze:
				self.maze, self.free_states = self._maze, self._free_states

			walk, _ = self.walk_agent(self.agents[agent_n], 0, stop=True)
			self.plot_walk(walk, ax=ax, alpha=0.85)
			walk_lengths.append(len(walk))

			ax.scatter(walk[0][0], walk[0][1], color="b", s=200)

		ax.scatter(self.goal_location[0], self.goal_location[1], color="r", s=200)
		

		print("mean length: ", np.mean(walk_lengths))

	def test_agent_on_arenas(self, agent_n, arena_type="open", maze=None):
		# copy
		self._maze = self.maze.copy()
		self._free_states = self.free_states.copy()

		# make new arena
		if arena_type == "open":
			self.maze = np.ones((self.grid_size, self.grid_size))
			self.free_states = [[i, j] for i in np.arange(self.grid_size) for j in np.arange(self.grid_size) if self.maze[i, j]==1]

		elif arena_type == "barrier":
			self.maze = np.ones((self.grid_size, self.grid_size))
			self.maze[int(self.grid_size/2):int(self.grid_size/2)+4, 
						int(self.grid_size/3): int(self.grid_size - self.grid_size/3)] = 0
			self.free_states = [[i, j] for i in np.arange(self.grid_size) for j in np.arange(self.grid_size) if self.maze[i, j]==1]

		elif arena_type == "halfcomplete_barrier":
			self.maze = np.ones((self.grid_size, self.grid_size))
			self.maze[int(self.grid_size/3):int(self.grid_size/3)+4, 0: int(self.grid_size - self.grid_size/4)] = 0
			self.free_states = [[i,j] for i in np.arange(self.grid_size) for j in np.arange(self.grid_size) if self.maze[i, j]==1]

		elif arena_type == "corridor":
			self.maze = np.zeros((self.grid_size, self.grid_size))
			x = int(self.grid_size/2)
			self.maze[:, x-10:x+11] = 1
			self.free_states = [[i,j] for i in np.arange(self.grid_size) for j in np.arange(self.grid_size) if self.maze[i, j]==1]

		elif arena_type == "maze":
			self.maze, self.free_states = self.get_maze_from_image(os.path.join(self.maze_models_folder, maze+".png"))
		else: raise NotImplementedError

		print(arena_type, " - free states: ", len(self.free_states))
		# self.test_image()

		# run the agent
		self.plot_agent_walk(agent_n)

		# restore maze
		self.maze, self.free_states = self._maze, self._free_states

	def test_image(self):
		img = np.zeros_like(self.maze)
		for x,y in self.free_states:
			img[x,y] = 1
		f,ax = plt.subplots()
		ax.imshow(img)


if 1 == 1:
	nnga = NNGaAnalyser()
	nnga.max_n_steps = 50
	nnga.n_walks= 5
	nnga.start_location = [19, 28]


	# ? Visualize agents
	# nnga.randomise_start_location = True
	# nnga.plot_all_agents_walks()
	# nnga.plot_agent_walk(-1)
	# nnga.test_agent_on_arenas(-1, arena_type="maze", maze="Square Maze")

	# ? test
	nnga.randomise_start_location = False
	# nnga.test_agent_on_arenas(-1, arena_type="open")
	# nnga.test_agent_on_arenas(-1, arena_type="corridor")
	# nnga.test_agent_on_arenas(-1, arena_type="barrier")

	# ? test on mazes
	nnga.randomise_start_location = True
	# nnga.test_agent_on_arenas(-1, arena_type="maze", maze="PathInt2")
	# nnga.test_agent_on_arenas(-1, arena_type="maze", maze="Square Maze")
	# nnga.test_agent_on_arenas(-1, arena_type="maze", maze="leftfar")
	nnga.test_agent_on_arenas(-1, arena_type="maze", maze="mazemodel")



# %%


# %%
# ! to run the learning
if __name__ == "__main__":
	ga = NNGa()
	ga.initialise_genomes()
	ga.initialise_agents()
	ga.evolve_maze()
	plt.show()
#%%


#%%
