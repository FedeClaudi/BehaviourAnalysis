import sys
sys.path.append('./')
import matplotlib.pyplot as plt
import numpy as np
from Processing.tracking_stats.math_utils import calc_distance_between_points_2d as dist
from math import exp

class Model:
    def __init__(self, env):
        self.env = env 
        self.actions()
        
        self.max_iters = 200
        self.max_steps = round(self.env.grid_size**2 / 2)

        # Parameters
        self.epsilon = .9
        self.alpha = 1
        self.gamma = .9
        self.move_away_incentive = .25  # reward for moving away from start in corresponding policy
        self.move_towards_incentive = .25  # reward for moving towards the shelter in the corresponding policy
        self.shelter_is_goal = [0, .5, 1]  # 0-1 probability of prefering the action that leads to the shelter

        self.no_change_thresh = 10

        Qs = [[[0,0,0,0] for i in range(self.env.grid_size)] for j in range(self.env.grid_size)]
        Qa = [[[0,0,0,0] for i in range(self.env.grid_size)] for j in range(self.env.grid_size)]
        Qsc = [[[0,0,0,0] for i in range(self.env.grid_size)] for j in range(self.env.grid_size)]
        self.Q = dict(shelter=Qs, away=Qa, shelter_close = Qsc)
        self.random_start = dict(shelter=True, away=False, shelter_close=True)
        self.walked = None

    def actions(self):
        self.actions = self.env.actions
        self.n_actions = len(self.actions.keys())-1
        self.actions_lookup = list(self.actions.values())


    def train(self):
        for (Q_name, Q), move_away_incentive in zip(self.Q.items(), [0, self.move_away_incentive, self.move_towards_incentive]):
            print("Learning Q: ", Q_name)
            n_actions = self.n_actions
            env = self.env
            not_change_count = 0
            self.traces = []

            for iter_n in np.arange(self.max_iters):
                if iter_n % 100 == 0: print(iter_n)
                env.reset(self.random_start[Q_name])
                game_over = False
                step = 0

                Q2 = Q.copy()
                trace = []
                while not (game_over or step > self.max_steps):
                    step += 1
                    curr_state = env.state()
                    trace.append(curr_state)

                    if np.random.rand() <= self.epsilon:  # epsilon-greedy policy
                        action = np.random.randint(0, self.n_actions)
                    else:
                        if(max(Q[curr_state[0]][curr_state[1]]) == min(Q[curr_state[0]][curr_state[1]])):
                            action = -1
                            # if Q[] function is unable to select action, then no action taken
                        else:
                            action = np.argmax(Q[curr_state[0]][curr_state[1]])
                            # best action from Q table
                    next_state, reward, game_over = env.act(action, move_away_incentive, Q_name)

                    # Q-learning update
                    Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + \
                        self.alpha*(reward + self.gamma*max(Q[next_state[0]][next_state[1]]) - Q[curr_state[0]][curr_state[1]][action])
            self.Q[Q_name] = Q


    def walk_step(self, goal, current, nxt):
        action_values = self.Q[goal][current[0]][current[1]]
        action = np.argmax(action_values)
        action = self.env.actions[action]

        if action == "down":
            nxt[1] -= 1
        elif action == "right":
            nxt[0] += 1
        elif action == "left":
            nxt[0] -= 1
        elif action == "up":
            nxt[1] += 1
        else:
            raise ValueError
        return nxt

    def walk(self):
        self.walked = {k:[] for k in self.Q.keys()}

        for Q_name, alternative_Q in self.Q.items():
            print("walking with alternative: ", Q_name)
            for prob_shelter in self.shelter_is_goal:
                curr = self.env.start.copy()
                step_n = 0
                max_steps = 500
                walked = []

                while curr != self.env.goal and step_n < max_steps:
                    step_n += 1
                    walked.append(curr.copy())

                    """ 
                        Compute what the next action would be given the goal of getting to the shelter as fast as possible
                    """
                    shelter_nxt = self.walk_step("shelter", curr, curr.copy())
                    

                    """ 
                        Compute what the next action would be given the alternative policy
                    """
                    alternative_nxt = self.walk_step(Q_name, curr, curr.copy())


                    """ 
                        Balance the two: select goal oriented action based on probability threshold
                    """
                    select =  np.random.uniform(0, 1, 1)
                    discounted_prob_shelter = (1 - prob_shelter) * exp(step_n/2)  # ? discount the alternative as time passess
                    if select >= discounted_prob_shelter:
                        nxt = shelter_nxt
                    else:
                        nxt = alternative_nxt


                    if nxt[0] < 0 or nxt[1]< 0 or nxt[0] > self.env.grid_size or nxt[1]>self.env.grid_size: break

                    if nxt in self.env.free_states: curr = nxt

                self.walked[Q_name].append(walked)
