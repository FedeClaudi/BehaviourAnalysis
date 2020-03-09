import numpy as np
import numpy.random as npr
from random import choice
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp


# Define distributions of path lengths and thethas
lengths_distributions = npr.uniform(25, 75, size=5000)
thethas_distributions = npr.uniform(25, 90, size=5000)

# Define Individual
def get_random_gene():
    return npr.uniform(-1, 1)

def get_mutated_gene(gene):
    return npr.normal(0, .1, 1) + gene
    

class Individual:
    speed = 1
    
    def __init__(self, idn):
        self.genome = [get_random_gene(), get_random_gene()]
        self.idn = idn

        self.reset_records()

    def reset_records(self):
        self.choices = []
        self.durations = []

    def fitness(self):
        return np.mean(self.durations)

    def __repr__(self):
        return f'Individual {self.idn} - fitness: ' + str(self.fitness())
    
    def __str__(self):
        return f'Individual {self.idn} - fitness: ' + str(self.fitness())

#  Define params
N_individuals = 1000
N_generation = 300
N_trials = 50

KEEP_BEST_perc = 33
KEEP_BEST = int((N_individuals/100)*KEEP_BEST_perc)


class Population:
    p_mutation = .025

    p_shortcut = [0, 0.1, 0.25, 0.5, 1]
    p_shortcut_idx = 0
    change_p_short_every = 100
    gen_num = 0

    def __init__(self):
        # Create population
        self.pop= []
        self.max_id = 0
        for i in range(N_individuals):
            self.add_to_population(Individual(self.max_id))

        # Create stats records
        self.stats = dict(
            mean_probability_correct = [],
            mean_escape_duration = [],
            mean_gene_0=[],
            mean_gene_1=[],
        )
        
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
       

        pool = mp.Pool(mp.cpu_count()-2)
        pool.map(self.run_individual, [(ind, p_shortcut) for ind in self.pop])
        pool.close()

        fitnesses = [ind.fitness() for ind in self.pop]
        sort_idx = np.argsort(fitnesses)
        self.pop = list(np.array(self.pop)[sort_idx])

    @staticmethod
    def run_individual(args):
        individual, p_shortcut = args
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
            choice_factor = individual.genome[0] * lf + individual.genome[1] * af

            # Choose an arm and compute escape duration
            if choice_factor < 0:
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


    def update_population(self):
        self.pop = self.pop[:KEEP_BEST]
        parent_gen = self.pop.copy()

        need_children = int((N_individuals - len(parent_gen))/2)
        for i in range(need_children):
            p1 = choice(parent_gen)
            p2 = choice(parent_gen)

            son1 = Individual(self.max_id)
            son1.genome[0] = p1.genome[0] if npr.random()>self.p_mutation else get_mutated_gene(p1.genome[0])
            son1.genome[1] = p2.genome[1] if npr.random()>self.p_mutation else get_mutated_gene(p2.genome[1])
            self.add_to_population(son1)

            son2 = Individual(self.max_id)
            son2.genome[0] = p2.genome[0] if npr.random()>self.p_mutation else get_mutated_gene(p2.genome[0])
            son2.genome[1] = p1.genome[1] if npr.random()>self.p_mutation else get_mutated_gene(p1.genome[1])
            self.add_to_population(son2)

        # reset stats kep by individuals
        for ind in self.pop:
            ind.reset_records()
        
    def update_stats(self):
        self.stats['mean_probability_correct'].append(np.nanmean([np.mean(i.choices) for i in self.pop]))
        self.stats['mean_escape_duration'].append(np.nanmean([i.fitness() for i in self.pop]))
        self.stats['mean_gene_0'].append(np.mean([i.genome[0] for i in self.pop]))
        self.stats['mean_gene_1'].append(np.mean([i.genome[1] for i in self.pop]))

    def evolve(self):
        for gen in tqdm(range(N_generation)):
            self.run_generation()
            self.update_stats()
            self.update_population()

    def plot_traces(self):
        f, axarr = plt.subplots(nrows=3, sharex=True, figsize=(16, 8))

        axarr[0].plot(self.stats['mean_probability_correct'])
        axarr[1].plot(self.stats['mean_gene_0'], label="Gene 0")
        axarr[1].plot(self.stats['mean_gene_1'], label="Gene 1")
        axarr[2].plot(self.stats['mean_escape_duration'])

        for ax in axarr:
            ax.legend()
            for v in [100, 200, 300]:
                ax.axvline(v-1)

        axarr[0].set(title='Mean p(correction)')
        _ = axarr[1].set(title='Mean genome', xlabel='# generation')
        _ = axarr[2].set(title='Mean escape duration')

        f.tight_layout()

if __name__ == '__main__':
    # Initialise individuals
    pop = Population()
    pop.evolve()
    pop.plot_traces()

    plt.show()


