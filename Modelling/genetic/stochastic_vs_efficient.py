import sys
sys.path.append('./')

import numpy as np
import random
from scipy.stats import bernoulli
import multiprocessing as mp

from Utilities.imports import *


"""
	Simulate the scenario in which agents can escape on two paths of different lengths in a probabilistic
	manner. p(death) is proportional to the length of each path and if theyir behaviour becomes too
	predictable they also die. 
"""

class Scene:
	# ! default params
	# ? maze setup
	left_l =  1
	right_l = 1

	# ? danger rates
	danger = .1 # probability of death for .1 unit length of path
	predator_memory = 800 # number of trials in the predator's memory 
	predator_risk_factor = .04 # reduce lethality of a predator that correctly anticipate the prey's behaviour
	
	# ? reproduction rates
	reproduction = .14 # probability that any agent will reproduce in a given generation
	mutation_rate = .001

	# ? Population params
	n_agents = 100
	max_agents = 200 
	n_generations = 400+1

	def __init__(self, verbose=False, **kwargs):
		self.set_params(**kwargs)
		self.verbose = verbose

		# initialise agents
		self.agents = [Agent(i, self, g0=True) for i in np.arange(self.n_agents)]

		# initialise predator
		self.predator = Predator(self)

		# Get risks associated with each path
		self.risks = {0:self.left_l*self.danger, 1:self.right_l*self.danger}
		for k,v in self.risks.items():
			if v > 1:
				raise ValueError("risk too large, p_death > 1, reduce danger")

	def set_params(self, **kwargs):
		self.left_l = kwargs.pop('left_l', self.left_l)
		self.right_l = kwargs.pop('right_l', self.right_l)
		self.danger = kwargs.pop('danger', self.danger)
		self.predator_memory = kwargs.pop('predator_memory', self.predator_memory)
		self.predator_risk_factor = kwargs.pop('predator_risk_factor', self.predator_risk_factor)
		self.reproduction = kwargs.pop('reproduction', self.reproduction)
		self.mutation_rate = kwargs.pop('mutation_rate', self.mutation_rate)
		self.n_agents = kwargs.pop('n_agents', self.n_agents)
		self.max_agents = kwargs.pop('max_agents', self.max_agents)
		self.n_generations = kwargs.pop('n_generations', self.n_generations)

	def get_params(self):
		params = {}
		params['left_l'] = self.left_l
		params['right_l'] = self.right_l
		params['danger'] = self.danger
		params['predator_memory'] = self.predator_memory
		params['predator_risk_factor'] = self.predator_risk_factor
		params['reproduction'] = self.reproduction
		params['mutation_rate'] = self.mutation_rate
		params['n_agents'] = self.n_agents
		params['max_agents'] = self.max_agents
		params['n_generations'] = self.n_generations
		return params


	def get_agents_ids(self):
		return [agent.id for agent in self.agents]

	def get_new_id(self):
		return np.max(self.get_agents_ids())+1

	def run(self):
		# Loop over generations
		self.traces = dict(pR=[], predator_bias=[], n_agents=[], deaths=[], kills=[], births=[])
		self.dead_agents = []

		for gennum in tqdm(np.arange(self.n_generations)):
			lefts, rights, deaths, kills = 0, 0, 0, 0
			survivors = []
			# Loop over agents 
			for agent in self.agents:
				# let agent decide
				choice, death = agent.trial()

				# Let predator learn
				self.predator.learn(choice)

				# evaluate
				if choice: rights += 1
				else: lefts += 1

				if not death: 
					# wasn't killed during escape, but maybe the predator anticipated it
					killed = self.predator.trial(choice)
					if killed:
						self.dead_agents.append(agent)
						kills += 1
					else:
						survivors.append(agent)
				else: 
					self.dead_agents.append(agent)
					deaths += 1
			self.agents = survivors

			# Loop over agents again, this time to reproduce
			if len(self.agents) <= self.max_agents:
				next_gen = []
				births = 0
				for agent in self.agents:
					if bernoulli.rvs(self.reproduction, size=1)[0]:
						partner = random.choice(self.agents)
						son = Agent(self.get_new_id(), self, parents=[agent, partner])
						next_gen.append(son)
						births += 1
				self.agents.extend(next_gen)

			# Summary
			if self.verbose:
				print("[{}] -- {} agents. {} deaths {} kills {} births".format(gennum, len(self.agents), deaths, kills, births))
			
			if len(self.agents):
				self.traces["pR"].append(rights/(rights+lefts))
				self.traces["n_agents"].append(len(self.agents))
				self.traces["deaths"].append(deaths)
				self.traces["kills"].append(kills)
				self.traces["births"].append(births)
				self.traces["predator_bias"].append(self.predator.risk)
			else:
				break
		self.traces = pd.DataFrame(self.traces)
		return self.traces, self.get_params()


	def plot_summary(self):
		# axes = self.traces.plot.line(subplots=True)
		# axes[0].axhline(0.5, color=black)
		# axes[0].set(ylim=[0, 1])
		# axes[1].axhline(0.5, color=black)
		# axes[1].set(ylim=[0, 1])
		# axes[-1].set(xlabel="generation #")

		f, ax = create_figure(subplots=False, facecolor=white, figsize=(16, 16))
		ax.plot(self.traces.pR, color=green, label="preys' pR")
		ax.plot(self.traces.predator_bias, color=red, label="predator pR")
		ax.axhline(0.5, color=black)
		ax.legend()
		ax.set(title="predator prey interaction", ylabel="p(R)", xlabel="# generations", ylim=[-0.1, 1.1])
		return ax


class Agent():
	n_genes = 10
	history = []

	def __init__(self, id, scene, g0 = False, parents=None):
		self.id = id
		self.scene = scene

		if g0:
			self.initialise_random()
		else:
			self.initialise_parents(*parents)

	def initialise_random(self):
		self.genome =  np.random.randint(0, 2, size=self.n_genes)

	def initialise_parents(self, p0, p1):
		# Get number of genes from p0
		n_genes = np.random.randint(0, self.n_genes+1, size=1)[0]
		self.genome = np.hstack([p0.genome[:n_genes], p1.genome[n_genes:]])
		if len(self.genome) != self.n_genes: raise ValueError("something went wrong during recombination")

	def trial(self):
		# choose left (0) or right (1) based on a p(R) = sum(genes)
		pR = np.sum(self.genome)/len(self.genome)
		choice = bernoulli.rvs(pR, size=1)[0]
		self.history.append(choice)

		# Evaluate probability of death by predation during escape
		p_death = self.scene.risks[choice]
		death = bernoulli.rvs(p_death, size=1)[0]

		return choice, death


class Predator():
	def __init__(self, scene):
		self.scene = scene

		# initialise random memory
		self.memory = list(np.random.randint(0, 2, size=self.scene.predator_memory))
	
	def learn(self, choice):
		if len(self.memory) > self.scene.predator_memory:
			self.memory = self.memory[1:]
		self.memory.append(choice)

	def estimate_danger(self):
		# p(R) over memory
		pR = np.mean(self.memory)
		self.risk = pR

	def trial(self, choice):
		# based on the choice of the prey, use the predator's memory to anticipate the prey's behaviour and kill it
		self.estimate_danger()

		if self.risk < .5 and not choice: # going left and correctly anticipated
			pass
		elif self.risk > .5 and choice: # going right and correctly anticipated
			pass
		else:
			return 0
		
		killed = bernoulli.rvs(self.scene.predator_risk_factor, size=1)[0]
		return killed




def run_scenes_in_parallel(scenes):
	# Define callback function to collect the output in `results`
	global results
	results = []
	def collect_result(result):
		global results
		results.append(result)

	if len(scenes) > mp.cpu_count(): print("Warning, too manu scenes for the number of CPUs")
	pool = mp.Pool(mp.cpu_count())

	print("Ready to run {} scenes in parallel".format(len(scenes)))
	for i, scene in enumerate(scenes):
		pool.apply_async(scene.run, args=(), callback=collect_result)
	pool.close()
	pool.join() # postpones the execution of next line of code until all processes in the queue are done.

	return results


if __name__ == "__main__":
	scenes = [Scene() for i in range(3)]
	res = run_scenes_in_parallel(scenes)