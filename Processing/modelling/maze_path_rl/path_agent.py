import sys
sys.path.append('./')
import matplotlib.pyplot as plt
import numpy as np
from Processing.tracking_stats.math_utils import calc_distance_between_points_2d as dist
from math import exp  
import json
import os
from random import choice

class Model:
    def __init__(self, env, load_trained=False):
        self.load_trained = load_trained
        self.env = env 
        self.actions()
        self.policies()
        
        self.max_iters = 201
        self.max_steps = round(self.env.grid_size**2 / 2)

        # Parameters
        self.epsilon = .95  # the higher the less greedy
        self.alpha = 1      # how much to value new experience, in a deterministic world set as 1
        self.gamma = .9     # discount on future rewards, the higher the less discount

        self.shelter_is_goal = [0, .5, 1]  # 0-1 probability of prefering the action that leads to the shelter


        self.shortest_walk = None

        self.n_random_walks = 10
        self.random_walks = []


    """
    -----------------------------------------------------------------------------------------------------------------
                        UTILS FUNCTIONS
    -----------------------------------------------------------------------------------------------------------------
    """

    def policies(self):
        # Get basic policies
        policies = ["shelter"] # , "away", "direct_vector", "intermediate", "combined")
        self.Q = {p:self.empty_policy() for p in policies}

    def empty_policy(self):
        return [[list(np.zeros(len(self.actions.keys()))) for i in range(self.env.grid_size)] for j in range(self.env.grid_size)]

    def actions(self):
        self.actions = self.env.actions
        self.n_actions = len(self.actions.keys())-1
        self.actions_lookup = list(self.actions.values())

    def save(self):
        # save the policies
        fld = "Processing/modelling/maze_path_rl/policies"
        for Q_name, Q in self.Q.items():
            savename = os.path.join(fld, self.env.name + "_" + Q_name+'.json')
            with open(savename, "w") as f:
                f.write(json.dumps(Q))

    def load(self):
        fld = "Processing/modelling/maze_path_rl/policies"
        for Q_name in self.Q.keys():
            loadname = os.path.join(fld, self.env.name + "_" + Q_name+'.json')
            Q = json.load(open(loadname))
            self.Q[Q_name] =  Q

    
    """
    -----------------------------------------------------------------------------------------------------------------
                        TRAINING
    -----------------------------------------------------------------------------------------------------------------
    """

    def train(self):
        if self.load_trained:
                try:
                    self.load()
                except:
                    pass # we gotsa train
                else:
                    return

        for Q_name, Q in self.Q.items():
            print("Learning Q: ", Q_name)
            n_actions = self.n_actions
            env = self.env
            not_change_count = 0
            self.traces = []

            # iterate the learning
            for iter_n in np.arange(self.max_iters):
                if iter_n % 100 == 0: print(iter_n)

                # reset starting position
                env.reset()

                # reset variables
                game_over = False
                step = 0
                Q2 = Q.copy()

                # Keep a trace of all action/states pairs
                trace = []

                # keep stepping and evaluating reward
                while not (game_over or step > self.max_steps):
                    step += 1
                    curr_state = env.state()
                    trace.append(curr_state)

                    if np.random.rand() <= self.epsilon:  # epsilon-greedy policy
                        action = np.random.randint(0, self.n_actions)
                    else:
                        action = np.argmax(Q[curr_state[0]][curr_state[1]])
                        # best action from Q table

                    # move in the environment
                    next_state, reward, game_over = env.act(action)

                    # Q-learning update
                    delta = self.gamma * max(Q[next_state[0]][next_state[1]]) - Q[curr_state[0]][curr_state[1]][action]
                    Q[curr_state[0]][curr_state[1]][action] = Q[curr_state[0]][curr_state[1]][action] + self.alpha*(reward + delta)
            
            self.Q[Q_name] = Q


    """
    -----------------------------------------------------------------------------------------------------------------
                        Walking functions
    -----------------------------------------------------------------------------------------------------------------
    """

    def step(self, policy, current, random_action = False):
        """[steps the agent while walking the maze. Given a certain policy and state (current position), selects the best 
        actions and moves the agent accordingly]
        
        Arguments:
            policy {[str]} -- [name of the self.Q entry - i.e. policy to sue while evaluating which action to take]
            current {[list]} -- [x,y coordinates of current position]
            random_action {[bool]} [if we should take a random action instead of using the policies to select an acyion]
        
        Returns:
            [type] -- [description]
        """

        # Select which action to perform
        if not random_action:
            action_values = self.Q[policy][current[0]][current[1]]
            action = np.argmax(action_values)
            action = self.env.actions[action]
        else:
            legal_actions = self.env.get_available_moves(current)
            action = "down"  # initialise action as down
            while "down" in action or "still" in action:  # dont move down silly agent [and dont stay still either!]
                action = choice(legal_actions)

        # Move the agent accordingly
        nxt = current.copy()
        up, down = -1, 1
        left, right = -1, 1
        if action == "down":
            nxt[1] += down
        elif action == "right":
            nxt[0] += right
        elif action == "left":
            nxt[0] += left
        elif action == "up":
            nxt[1] += up
        elif action == "up-right":
            nxt[0] += right
            nxt[1] += up
        elif action == "up-left":
            nxt[0] += left
            nxt[1] += up
        elif action == "down-right":
            nxt[0] += right
            nxt[1] += down
        elif action == "down-left":
            nxt[0] += left
            nxt[1] += down

        return nxt

    def shortest_walk_f(self,  start=None):
        """[Using the sheler policy, it finds the shortest path to the shelter]
        

        Keyword Arguments:
            start {[list]} -- [alternative start to default one, optional] (default: {None})
        
        Returns:
            [list] -- [list of x,y coordinates at each step n the walk]
        """

        walk = []
        if start is None:
            curr = self.env.start.copy()
        else:
            curr = start.copy()

        step_n = 0
        # do each step
        while curr != self.env.goal and step_n < self.max_steps:
            step_n += 1
            walk.append(curr.copy())

            nxt = self.step("shelter", curr)

            # Check that nxt is a legal move
            if nxt[0] < 0 or nxt[1] < 0 or nxt[0] > self.env.grid_size or nxt[1] > self.env.grid_size:
                break
            if nxt in self.env.free_states:
                curr = nxt
        walk.append(curr)

        self.shortest_walk = walk
        return walk

    def random_walk(self, n_steps):
        walk = []
        step_n = 0
        curr = self.env.start.copy()

        while step_n <= n_steps and curr != self.env.goal:
            step_n += 1
            walk.append(curr.copy())
            nxt = self.step("shelter", curr, random_action=True)
            # Check that nxt is a legal move
            # if nxt[0] < 0 or nxt[1] < 0 or nxt[0] > self.env.grid_size or nxt[1] > self.env.grid_size:
            #     break
            # if nxt in self.env.free_states:
            #     curr = nxt
            curr = nxt
        walk.append(curr)
        return walk


    def random_walks_f(self):
        """
            This function does N random walks from start to goal without using any policy.
            Each walk can be max M steps long where M = 3 * length of the shortest goal.
            After X steps the walk is not random anymore and the shelter policy is used to reach the shelter.
            X = .5 * length of the shortest path to the goal
            Only walks that reach the shelter are kept
        """
        print("Random walks time")

        if self.shortest_walk == None: self.shortest_walk_f()

        # Define params
        max_n_steps = 3* len(self.shortest_walk)
        n_random_steps = round(len(self.shortest_walk) * .75 )

        # Keep looping until we have enough walks
        counter = 0
        while len(self.random_walks) < self.n_random_walks:
            counter += 1
            # do the random part of the walk
            walk = self.random_walk(n_random_steps)
            stopped_at = walk[-1]

            # Continue the walk using the policy
            walk.extend(self.shortest_walk_f(start=stopped_at))

            # If the walk isn't too long, append it to the list
            if len(walk) <= max_n_steps:
                self.random_walks.append(walk)

        print("{} out of {} walks were short enough".format(self.n_random_walks, counter))

        # Plot the walks
        f, ax = plt.subplots()
        self.get_maze_image()
        ax.imshow(self.maze, interpolation='none', cmap='gray')
        for w in self.random_walks:
            ax.scatter([x for x,y in w], [y for x,y in w], alpha=.5)
            

        plt.show()
        a = 1

    """
    -----------------------------------------------------------------------------------------------------------------
                        Plotting functions
    -----------------------------------------------------------------------------------------------------------------
    """

    def plot_single_walk(self, w):
        f, ax = plt.subplots()
        self.get_maze_image()
        ax.imshow(self.maze, interpolation='none', cmap='gray')
        ax.scatter([x for x,y in w], [y for x,y in w], alpha=.5)

    def get_maze_image(self):
        Q = self.Q['shelter']
        free = self.env.free_states
        grid_size = self.env.grid_size
        pol = [[max(Q[i][j]) for i in range(grid_size)] for j in range(grid_size)]
        self.maze =  [[1 if [i, j] in free else 0 for i in range(grid_size)] for j in range(grid_size)]

    def policy_plot(self):

        Q = self.Q
        start = self.env.start
        free = self.env.free_states

        
        grid_size = self.env.grid_size
        name = self.env.name

        f, axarr = plt.subplots(ncols=3, nrows=len(list(self.Q.keys())), figsize=(16, 12))
        axarr = axarr.reshape(1, 3)

        for i, (Q_name, Q) in enumerate(self.Q.items()):
            if not "subgoal" in Q_name:
                goal = self.env.goal
            else:
                n = int(Q_name.split("_")[-1])
                goal = self.env.subgoals[n]


            grid_size = len(Q)
            pol = [[max(Q[i][j]) for i in range(grid_size)] for j in range(grid_size)]
            maze = [[1 if [i, j] in free else 0 for i in range(grid_size)] for j in range(grid_size)]
            act = [[np.argmax(Q[i][j]) if pol[j][i] > 0 else np.nan for i in range(grid_size)] for j in range(grid_size)]


            axarr[i, 0].imshow(maze, interpolation='none', cmap='gray')
            axarr[i, 0].scatter(start[0], start[1], c='g', s = 50)
            axarr[i, 0].scatter(goal[0], goal[1], c='b', s=50)

            walk = self.shortest_walk_f()
            x, y = [x for x, y in walk], [y for x, y in walk]
            axarr[i, 0].scatter(x, y, c='r', alpha=0.5, s=30)

            axarr[i, 1].imshow(act, interpolation='none',  cmap="tab20")
            axarr[i, 2].imshow(pol, interpolation='none')

            axarr[i, 0].set(title=list(self.Q.keys())[i])
            axarr[i, 1].set(title="actions")
            axarr[i, 2].set(title="Policy")

        axarr = axarr.flatten()
        for ax in axarr:
            ax.set(xticks=[], yticks=[])
        f.tight_layout()
        f.savefig("Processing/modelling/maze_path_rl/results/{}.png".format(name))



"""
-----------------------------------------------------------------------------------------------------------------
                    Obsolete functions
-----------------------------------------------------------------------------------------------------------------
"""

"""
    process_walks given a list of walks measures stuff like walk length, total offest from direct vector...

"""


# def process_walks(self, walks):
#     def normalise_by_rightmedium(vector, rmidx):
#         return [x/vector[rmidx] for x in vector]

#     # ! walks.append((walk, distance_travelled, tot_angle, at_subgoal))


#     # Get the lengths of the direct path between start and goal
#     direct_vector_distance = calc_distance_between_points_2d(self.learner.env.start, self.learner.env.goal)

#     sort_idxs, labels = self.path_order_lookup(self.learner.env.name)
#     right_medium_idx = labels.index("Right_Medium")
#     walks = [walks[i] for i in sort_idxs]

#     # Get the normalised length of each path
#     walk_lengths = [w[1] for w in walks]
#     normalised_lengths = normalise_by_rightmedium(walk_lengths, right_medium_idx)

#     # Get distance at subgoal from direct path
#     position_at_subgoals = [np.array(w[0][w[3]]) for w in walks]
#     direct = [np.array(self.learner.env.start), np.array(self.learner.env.goal)]
#     distance_at_subgoals = [abs(calc_distane_between_point_and_line(direct, p)) for p in position_at_subgoals]
#     normalised_distance_at_sg = normalise_by_rightmedium(distance_at_subgoals, right_medium_idx)

#     # Get tot angle off direct path until subgoal
#     angles = [w[2] for w in walks]
#     normalised_angles = normalise_by_rightmedium(angles, right_medium_idx)

#     # Plot
#     f, axarr = plt.subplots(nrows=3, figsize=(12, 9))
#     xx = np.arange(len(walk_lengths))
#     colors = self.colors_lookup(labels)

#     axarr[0].bar(xx, normalised_lengths, color=colors)
#     axarr[0].axhline(1, linestyle="--", color="k")
#     axarr[0].set(title="normalised path length", ylabel="n steps.", xticks=xx, xticklabels=labels)

#     axarr[1].bar(xx, normalised_distance_at_sg, color=colors)
#     axarr[1].axhline(1, linestyle="--", color="k")
#     axarr[1].set(title="Normalised distance from direct path at subgoal", ylabel="px", xticks=xx, xticklabels=labels)

#     axarr[2].bar(xx, normalised_angles, color=colors)
#     axarr[2].axhline(1, linestyle="--", color="k")
#     axarr[2].set(title="normalised Angle off direct until subgoal", ylabel="comulative theta", xticks=xx, xticklabels=labels)


#     f.tight_layout()
#     f.savefig("Processing/modelling/maze_path_rl/results/{}_pathsvars.png".format(self.learner.env.name))

    
# @staticmethod
# def path_order_lookup(exp):
#     if exp == "PathInt":
#         return [2, 1, 0], ["Left_Far", "Centre", "Right_Medium"]
#     elif exp == "PathInt2":
#         return [1, 0], ["Left_Far", "Right_Medium"]

# @staticmethod
# def colors_lookup(labels):
#     colors = dict(
#         Left_Far = [.2, .6, .2],
#         Left_Medium = [.2, .6, .2],
#         Centre = [.2, .2, .6],
#         Right_Medium = [.6, .2, .2]
#     )
#     return [colors[l] for l in labels]