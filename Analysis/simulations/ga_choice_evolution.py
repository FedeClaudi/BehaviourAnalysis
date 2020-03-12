# %%
import numpy as np
import numpy.random as npr
from random import choice
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from fcutils.maths.geometry import calc_distance_between_points_2d, calc_angle_between_vectors_of_points_2d



# %%
def coin_toss(th = 0.5):
    if npr.random()>th:
        return True
    else:
        return False

class Maze:
    def __init__(self, A, B, C_l, C_r):
        self.A = A
        self.B = B
        self.C_l = C_l
        self.C_r = C_r

        self.compute_sides()
        self.compute_xhat()

    def compute_sides(self):
        self.AB =   calc_distance_between_points_2d(self.A, self.B)

        self.AC_l =     calc_distance_between_points_2d(self.A, self.C_l)
        self.C_lB =     calc_distance_between_points_2d(self.C_l, self.B)
        self.AC_lB = self.AC_l + self.C_lB

        self.AC_r =     calc_distance_between_points_2d(self.A, self.C_r)
        self.C_rB =     calc_distance_between_points_2d(self.C_r, self.B)
        self.AC_rB = self.AC_r + self.C_rB

    def compute_xhat(self, niters=100):
        self.xhat_l = 0
        for i in range(niters):
            P = self.get_P(shortcut_on='left')
            AP = calc_distance_between_points_2d(self.A, P)
            PB = calc_distance_between_points_2d(P, self.B)
            self.xhat_l += AP + PB

            # if AP + PB > self.AC_lB:
            #     raise ValueError
        self.xhat_l = self.xhat_l/niters



        self.xhat_r = 0
        for i in range(niters):
            P = self.get_P(shortcut_on='right')
            AP = calc_distance_between_points_2d(self.A, P)
            PB = calc_distance_between_points_2d(P, self.B)
            self.xhat_r += AP + PB

            # if AP + PB > self.AC_rB:
            #     raise ValueError
        self.xhat_r = self.xhat_r/niters



    def compute_xbar(self, p_short):
        if p_short < 0 or p_short > 1: raise ValueError
        xbar_l = (1-p_short)*self.AC_lB + p_short*self.xhat_l
        xbar_r = (1-p_short)*self.AC_rB + p_short*self.xhat_r
        return xbar_l, xbar_r

    def get_P(self, shortcut_on='left'):
        # Choose between left and right arm 
        if shortcut_on == 'left':
            ac = self.AC_l
            cb = self.C_lB
            c = self.C_l
            self.shortcut_on = 'left'
        else:
            ac = self.AC_r
            cb = self.C_rB
            c = self.C_r
            self.shortcut_on = 'right'

        # See if P is in AC or CB
        segments_ratio = ac/(cb + ac)
        if coin_toss(th=segments_ratio): # P appars in CL
            self.P = (npr.uniform(c[0], self.B[0]), npr.uniform(c[1], self.B[1]))
        else:
            self.P = (npr.uniform(self.A[0], c[0]), npr.uniform(self.A[1], c[1]))
        return self.P


class Environment:
    def __init__(self, **kwargs):
        self.p_short = kwargs.pop('p_short', .1)
        self.A = kwargs.pop('A', (0, 0)) # threat pos
        self.B = kwargs.pop('B', (0, 10)) # shelter pos

        self.N_mazes = kwargs.pop('N_mazes', 10)
        self.N_generations = kwargs.pop('N_generations', 50)
        self.N_agents = kwargs.pop('N_agents', 50)
        self.keep_top_perc = kwargs.pop('keep_top_perc', 33)

        self.x_minmax = kwargs.pop('x_minmax', 6)

        self.get_mazes()

    def get_mazes(self):
        self.mazes = []
        for i in np.arange(self.N_mazes):
            C_l = (npr.uniform(self.A[0]-self.x_minmax, self.A[0]), npr.uniform(self.A[1], self.B[1]))
            C_r = (npr.uniform(self.A[0], self.A[0]+self.x_minmax), npr.uniform(self.A[1], self.B[1]))

            self.mazes.append(Maze(self.A, self.B, C_l, C_r))

    def run_trial(self, agent, maze):
        # Get the agent's choice
        agent_choice = agent.choose(maze)

        # Get the actual path lengths
        xbar_l, xbar_r = maze.compute_xbar(self.p_short)

        # Evaluate outcome
        if agent_choice == 'left':
            agent.outcomes.append(xbar_l)
        else:
            agent.outcomes.append(xbar_r)



class Agent:
    def __init__(self, p_short=None):
        if p_short is None:
            self.p_short = npr.uniform(0, 1)
        else:
            self.p_short = p_short

        if self.p_short > 1: self.p_short = 1
        if self.p_short < 0: self.p_short = 0

        self.outcomes = []

    def choose(self, maze):
        xbar_l, xbar_r = maze.compute_xbar(self.p_short)

        # choose according to the ratio
        th = xbar_l/(xbar_l + xbar_r)
        if coin_toss(th=1 - th):
            choice = 'left'
        else:
            choice = 'right'

        return choice

    def compute_fitness(self):
        self.fitness = np.mean(self.outcomes)


    

class Population(Environment):
    def __init__(self, **kwargs):
        Environment.__init__(self, **kwargs)

        self.gen_num = 0
        self.agents = [Agent() for i in range(self.N_agents)]

        self.keep_top = np.int((self.N_agents/100)*self.keep_top_perc)

        self.stats = dict(
            world_p_short = [],
            agents_p_short = [],
        )

    def run_generation(self):
        if self.gen_num % 10 == 0:
            self.get_mazes()

        for agent in self.agents:
            for maze in self.mazes:
                self.run_trial(agent, maze)
            agent.compute_fitness()
        self.gen_num += 1

    def update_population(self):
        # keep best
        pop_fitness = [a.fitness for a in self.agents]
        sort_idx = np.argsort(pop_fitness)
        self.agents = list(np.array(self.agents)[sort_idx])[:self.keep_top]

        # replenish population
        prev_gen = self.agents.copy()
        while len(self.agents) < self.N_agents:
            # choose a random parent
            parent = choice(prev_gen)
            self.agents.append(Agent(parent.p_short + npr.normal(0, .05)))

    def update_stats(self):
        self.stats['world_p_short'].append(self.p_short)
        self.stats['agents_p_short'].append(np.mean([a.p_short for a in self.agents]))

    def plot(self):
        f, ax = plt.subplots(figsize=(12, 8))

        for k,v in self.stats.items():
            ax.plot(v, label=k)
        ax.legend()
        ax.set(xlabel='# generations', ylabel='probability shortcut')

    def evolve(self):
        for gen_n in tqdm(range(self.N_generations)):
            self.update_stats()
            self.run_generation()
            self.update_population()


# %%
pop = Population(N_generations=750, N_agents=100, keep_top_perc=50, p_short=0)
pop.evolve()

# pop.p_short=.0
# pop.evolve()


pop.plot()




# %%
