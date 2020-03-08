# %%
import numpy as np
import numpy.random as npr
from random import choice
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%
# Define distributions of path lengths and thethas
lengths_distributions = npr.uniform(25, 75, size=5000)
thethas_distributions = npr.uniform(25, 90, size=5000)

# %%
# Define Individual
class Individual:
    speed = 10
    
    def __init__(self, idn):
        self.genome = [npr.uniform(-3, 3), npr.uniform(-3, 3)]

        self.choices = []
        self.durations = []

        self.idn = idn

    def fitness(self):
        return np.mean(self.durations)

    def __repr__(self):
        return f'Individual {self.idn} - fitness: ' + str(self.fitness())
    
    def __str__(self):
        return f'Individual {self.idn} - fitness: ' + str(self.fitness())

#  Define params
N_individuals = 500
N_generation = 100
N_trials = 40

KEEP_BEST_perc = 33
KEEP_BEST = int((N_individuals/100)*KEEP_BEST_perc)


class Population:
    keep_best = KEEP_BEST
    p_mutation = .1

    p_shortcut = [0.0, 0, 0, 0, 0, 0]#, 0.01, 0.05, .1, .5]
    p_shortcut_idx = 0
    change_p_short_every = 100
    gen_num = 0

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
        self.gen_num += 1

        if self.gen_num > 0:
            if self.gen_num % self.change_p_short_every == 0:
                self.p_shortcut_idx +=1 
        p_shortcut = self.p_shortcut[self.p_shortcut_idx]
       

        for individual in self.pop:
            for trial in range(N_trials):
                # ENVIRONMENT COMPUTATION
                left = [choice(lengths_distributions), choice(thethas_distributions)]
                right = [choice(lengths_distributions), choice(thethas_distributions)]

                # apply shortcut
                if npr.random() < p_shortcut:
                    # only apply to one arm
                    if npr.random() < .5:
                        left[0] = left[0] - left[0]*np.cos(np.radians(left[1]))
                    else:
                        right[0] = right[0] - right[0]*np.cos(np.radians(right[1]))

                # lengths factor
                lf = (left[0] - right[0])/(left[0] + right[0])  # Will be positive if left longer right

                # Angles factor
                af = (left[1] - right[1])/(left[1] + right[1]) # Will be positive if theta left > theta right

                # AGENT COMPUTATIONS
                # Compute choice
                choice_factor = individual.genome[0] * lf #+ individual.genome[1] * af

                # rescale to 0-1 range
                choice_factor = (choice_factor + 1)/2
                # TODO The probel is with this normalization, it doesn't work for all gene values
                
                # Choose an arm and compute escape duration
                if npr.random() < choice_factor:
                    chosen = 'r'
                    path = right
                else:
                    chosen = 'l'
                    path = left

                if left[0] < right[0]:
                    if chosen == 'l':
                        individual.choices.append(1)
                    else:
                        individual.choices.append(0)
                elif left[0] > right[0]:
                    if chosen == 'r':
                        individual.choices.append(1)
                    else:
                        individual.choices.append(0)
                else:
                    individual.choices.append(1)

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
            son1.genome[0] = p1.genome[0] if npr.random()>self.p_mutation else npr.uniform(-3, 3)
            son1.genome[1] = p2.genome[1] if npr.random()>self.p_mutation else npr.uniform(-3, 3)
            self.add_to_population(son1)

            son2 = Individual(self.max_id)
            son2.genome[0] = p2.genome[0] if npr.random()>self.p_mutation else npr.uniform(-3, 3)
            son2.genome[1] = p1.genome[1] if npr.random()>self.p_mutation else npr.uniform(-3, 3)
            self.add_to_population(son2)
        
    def evolve(self):
        mean_p_correct = []
        mean_fitness = []
        best_individual = []
        best_individual_p_correct = []

        mean_gene_0, mean_gene_1 = [], []

        for gen in tqdm(range(N_generation)):
            mean_fitness.append(np.nanmean([i.fitness() for i in self.pop]))
            mean_p_correct.append(np.nanmean([np.mean(i.choices) for i in self.pop]))
            self.run_generation()
            self.update_population()

            best_individual.append(self.pop[0])
            best_individual_p_correct.append(np.nanmean(self.pop[0].choices))
            mean_gene_0.append(np.mean([i.genome[0] for i in self.pop]))
            mean_gene_1.append(np.mean([i.genome[1] for i in self.pop]))

        return mean_fitness, mean_p_correct, best_individual, best_individual_p_correct, \
                            np.array(mean_gene_0), np.array(mean_gene_1)



# %%
# Initialise individuals
pop = Population()
fitness, mean_p_correct, individuals, best_individual_p_correct, mean_gene_0, mean_gene_1 = pop.evolve()

# Summary plots

f, axarr = plt.subplots(nrows=5, sharex=True, figsize=(16, 10))

axarr[0].plot(fitness, label='Mean fitness')

axarr[1].plot([ind.genome[0] for ind in individuals], label="Gene 0")
axarr[1].plot([ind.genome[1] for ind in individuals], label="Gene 1")

axarr[2].plot(mean_gene_0, label="Gene 0")
axarr[2].plot(mean_gene_1, label="Gene 1")

axarr[3].plot((mean_gene_0-mean_gene_1)/(mean_gene_0+mean_gene_1), label="Gene 0/gene 1")

axarr[4].plot(best_individual_p_correct, label='Best')
axarr[4].plot(mean_p_correct, label='Mean')

for ax in axarr:
    ax.legend()
    for v in [100, 200, 300]:
        ax.axvline(v)

axarr[0].set(title='Mean fitness')
_ = axarr[1].set(title='Best genome')
_ = axarr[2].set(title='Mean genome', xlabel='# generation')
_ = axarr[3].set(title='Gene 0 / Gene 1')
_ = axarr[4].set(title='Mean p(correct) [best individual]')

f.tight_layout()

# %%
