import sys
sys.path.append('./')
import matplotlib.pyplot as plt
import numpy as np
from Processing.tracking_stats.math_utils import calc_distance_between_points_2d as dist
from math import exp
import json
import os

class Model:
    def __init__(self, env, load_trained=False):
        self.load_trained = load_trained
        self.env = env 
        self.actions()
        
        self.max_iters = 201
        self.max_steps = round(self.env.grid_size**2 / 2)

        # Parameters
        self.epsilon = .9
        self.alpha = 1
        self.gamma = .8

        self.move_away_incentive = .25  # reward for moving away from start in corresponding policy
        self.move_towards_incentive = .25  # reward for moving towards the shelter in the corresponding policy
        self.shelter_is_goal = [0, .5, 1]  # 0-1 probability of prefering the action that leads to the shelter

        Qs = [[list(np.zeros(self.n_actions)) for i in range(self.env.grid_size)] for j in range(self.env.grid_size)]
        Qa = [[list(np.zeros(self.n_actions)) for i in range(self.env.grid_size)] for j in range(self.env.grid_size)]
        Qsc = [[list(np.zeros(self.n_actions)) for i in range(self.env.grid_size)] for j in range(self.env.grid_size)]
        self.Q = dict(shelter=Qs, away=Qa, direct_vector = Qsc)
        self.random_start = dict(shelter=True, away=False, direct_vector=True)
        self.walked = None

    def actions(self):
        self.actions = self.env.actions
        self.n_actions = len(self.actions.keys())-1
        self.actions_lookup = list(self.actions.values())


    def train(self):
        if self.load_trained:
                try:
                    self.load()
                except:
                    pass # we gotsa train
                else:
                    return

                    
        for (Q_name, Q), move_away_incentive in zip(self.Q.items(), [0, self.move_away_incentive, self.move_towards_incentive]):
            
            print("Learning Q: ", Q_name)
            n_actions = self.n_actions
            env = self.env
            not_change_count = 0
            self.traces = []

            for iter_n in np.arange(self.max_iters):
                if iter_n % 100 == 0: print(iter_n)

                # Start from random location 
                env.reset(self.random_start[Q_name])

                # reset variables
                game_over = False
                step = 0
                Q2 = Q.copy()

                # Keep a trace of all action/states pairs
                trace = []

                while not (game_over or step > self.max_steps):
                    step += 1
                    curr_state = env.state()
                    trace.append(curr_state)

                    if np.random.rand() <= self.epsilon:  # epsilon-greedy policy
                        action = np.random.randint(0, self.n_actions)
                    else:
                        action = np.argmax(Q[curr_state[0]][curr_state[1]])
                        # best action from Q table

                    next_state, reward, game_over = env.act(action, move_away_incentive, Q_name,)

                    # Q-learning update
                    delta = self.gamma * \
                        max(Q[next_state[0]][next_state[1]]) - \
                        Q[curr_state[0]][curr_state[1]][action]
                    Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + \
                        self.alpha*(reward + delta)
            self.Q[Q_name] = Q

    def save(self):
        # save the policies
        fld = "Processing/modelling/maze_path_rl/policies"
        for Q_name, Q in self.Q.items():
            savename = os.path.join(fld, Q_name+'json')
            with open(savename, "w") as f:
                f.write(json.dumps(Q))

    def load(self):
        fld = "Processing/modelling/maze_path_rl/policies"
        for Q_name in self.Q.keys():
            loadname = os.path.join(fld, Q_name+'json')
            Q = json.load(open(loadname))
            self.Q[Q_name] =  Q
