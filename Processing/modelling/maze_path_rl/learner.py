import matplotlib.pyplot as plt
import numpy as np



class Model:
    def __init__(self, env):
        self.env = env 
        self.actions()
        
        self.max_iters = 500
        self.max_steps = round(self.env.grid_size**2 / 2)


        self.epsilon = .9
        self.alpha = 1
        self.gamma = .9

        self.no_change_thresh = 10

        self.Q = [[[0,0,0,0] for i in range(self.env.grid_size)] for j in range(self.env.grid_size)]#
        self.walked = None

    def actions(self):
        self.actions = self.env.actions
        self.n_actions = len(self.actions.keys())-1


    def train(self):
        n_actions = self.n_actions
        env = self.env
        not_change_count = 0
        Q = self.Q
        self.traces = []

        for iter_n in np.arange(self.max_iters):
            if iter_n % 10 == 0: print(iter_n)
            env.reset()
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
                next_state, reward, game_over = env.act(action)
                # Q-learning update
                Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + \
                    self.alpha*(reward + self.gamma*max(Q[next_state[0]][next_state[1]]) - Q[curr_state[0]][curr_state[1]][action])
            self.traces.append(trace)
        self.Q = Q

    def plot_traces(self):
        plt.figure()

        for trace in self.traces:
            plt.plot([x for (x,y) in trace], [y for (x,y) in trace], alpha=.1)

        plt.show()


    def change(self, Q1, Q2):
        thres = 0.0 
        for i in self.env.free_states:
            prev_val = sum(Q1[i[0]][i[1]])
            new_val = sum(Q2[i[0]][i[1]])
            if(abs(prev_val - new_val) > thres):
                change = 1
                break
            else:
                change = 0
        return change


    def walk(self):
        curr = self.env.start.copy()
        step_n = 0
        max_steps = 500
        self.walked = []
        while curr != self.env.goal and step_n < max_steps:
            step_n += 1

            self.walked.append(curr.copy())

            try:
                action_values = self.Q[curr[0]][curr[1]]
            except:
                break
            action = np.argmax(action_values)
            action = self.env.actions[action]

            if action == "down":
                curr[1] -= 1   
            elif action == "right":
                curr[0] += 1   
            elif action == "left":
                curr[0] -= 1
            elif action == "up":
                curr[1] += 1
            else:
                raise ValueError

            if curr[0] < 0 or curr[1]< 0 or curr[0] > self.env.grid_size or curr[1]>self.env.grid_size: break