# %%
import numpy as np
import numpy.random as npr
from random import choice
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%
#  Define params
N_individuals = 1000
N_generation = 200
N_trials = 20

KEEP_BEST_perc = 50
KEEP_BEST = int((N_individuals/100)*KEEP_BEST_perc)

# %%
# Define distributions of path lengths and thethas
lengths_distributions = npr.uniform(25, 75, size=5000)
thethas_distributions = npr.uniform(25, 90, size=5000)

# %%
# Define Individual
class Individual:
    speed = 10
    
    def __init__(self, idn):
        self.genome = [npr.uniform(-1, 1), npr.uniform(-1, 1)]

        self.choices = []
        self.durations = []

        self.idn = idn

    def fitness(self):
        return np.mean(self.durations)

    def __repr__(self):
        return f'Individual {self.idn} - fitness: ' + str(self.fitness())
    
    def __str__(self):
        return f'Individual {self.idn} - fitness: ' + str(self.fitness())


class Population:
    keep_best = KEEP_BEST
    p_mutation = .05

    inverted = False

    def __init__(self):
        self.pop= []

        self.max_id = 0
        for i in range(N_individuals):
            self.add_to_population(Individual(self.max_id))
        
    def __repr__(self):
        return f'Population with {len(self.pop)} individuals'
    
    def __str__(self):
        return f'Population with {len(self.pop)} individuals'

    def add_to_population(self, individual):
        self.pop.append(individual)
        self.max_id += 1

    def run_generation(self):
        for individual in self.pop:
            for trial in range(N_trials):
                # ENVIRONMENT COMPUTATION
                left = (choice(lengths_distributions), choice(thethas_distributions))
                right = (choice(lengths_distributions), choice(thethas_distributions))

                # lengths factor
                lf = (left[0] - right[0])/(left[0] + right[0])  # Will be positive if left longer right

                # Angles factor
                af = (left[1] - right[1])/(left[1] + right[1]) # Will be positive if theta left > theta right

                # AGENT COMPUTATIONS
                # Compute choice
                if not self.inverted:
                    choice_factor = individual.genome[0] * lf + individual.genome[1] * af
                else:
                    choice_factor = individual.genome[1] * lf + individual.genome[0] * af

                # rescale to 0-1 range
                choice_factor = (choice_factor + 1)/2
                
                # Choose an arm and compute escape duration
                if npr.random() < choice_factor:
                    individual.choices.append('r')
                    path = right
                else:
                    individual.choices.append('l')
                    path = left

                individual.durations.append(path[0]/individual.speed)

        fitnesses = [ind.fitness() for ind in self.pop]

        sort_idx = np.argsort(fitnesses)
        self.pop = list(np.array(self.pop)[sort_idx])[::-1]

    def update_population(self):
        self.pop = self.pop[:self.keep_best]
        parent_gen = self.pop.copy()

        need_children = int((N_individuals - len(parent_gen))/2)
        for i in range(need_children):
            p1 = choice(parent_gen)
            p2 = choice(parent_gen)

            son1 = Individual(self.max_id)
            son1.genome[0] = p1.genome[0] if npr.random()>self.p_mutation else npr.uniform(-1, 1)
            son1.genome[1] = p2.genome[1] if npr.random()>self.p_mutation else npr.uniform(-1, 1)
            self.add_to_population(son1)

            son2 = Individual(self.max_id)
            son2.genome[0] = p2.genome[0] if npr.random()>self.p_mutation else npr.uniform(-1, 1)
            son2.genome[1] = p1.genome[1] if npr.random()>self.p_mutation else npr.uniform(-1, 1)
            self.add_to_population(son2)
        
    def evolve(self):
        max_fitness = []
        best_individual = []
        mean_gene_0, mean_gene_1 = [], []

        for gen in tqdm(range(N_generation)):
            self.run_generation()
            self.update_population()

            best_individual.append(self.pop[0])
            max_fitness.append(self.pop[0].fitness())
            mean_gene_0.append(np.mean([i.genome[0] for i in self.pop]))
            mean_gene_1.append(np.mean([i.genome[1] for i in self.pop]))

            if gen == 50:
                self.inverted = True
            if gen == 100:
                self.inverted = False
        return max_fitness, best_individual, mean_gene_0, mean_gene_1



# %%
# Initialise individuals
pop = Population()
fitness, individuals, mean_gene_0, mean_gene_1 = pop.evolve()



# Summary plots

f, axarr = plt.subplots(nrows=3, sharex=True)

axarr[0].plot(fitness, label='Max fitness')

axarr[1].plot([ind.genome[0] for ind in individuals], label="Gene 0")
axarr[1].plot([ind.genome[1] for ind in individuals], label="Gene 1")

axarr[2].plot(mean_gene_0, label="Gene 0")
axarr[2].plot(mean_gene_1, label="Gene 1")


axarr[0].legend()
axarr[1].legend()
axarr[2]

axarr[0].axvline(50)
axarr[1].axvline(50)
axarr[2].axvline(50)

axarr[0].axvline(100)
axarr[1].axvline(100)
axarr[2].axvline(100)

axarr[0].set(title='Max fitness')
_ = axarr[1].set(title='Best genome')
_ = axarr[2].set(title='Mean genome', xlabel='# generation')




# %%
