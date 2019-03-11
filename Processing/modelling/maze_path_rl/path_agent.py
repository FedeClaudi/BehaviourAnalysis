import sys
sys.path.append('./')
import matplotlib.pyplot as plt
import numpy as np
from Processing.tracking_stats.math_utils import calc_distance_between_points_in_a_vector_2d as dist
from Processing.tracking_stats.math_utils import get_n_colors, calc_angle_between_points_of_vector, calc_ang_velocity, line_smoother
from math import exp  
import json
import os
from random import choice
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.signal import resample
import random
import pickle




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

        self.n_random_walks = 1500
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
            # Select an action based on each action's probability
            action_values = self.Q[policy][current[0]][current[1]]
            action_values = action_values / np.sum(action_values)   # normalise
            selected_action = np.random.choice(action_values, 1, p=action_values)
            action_id = random.choice(np.where(action_values == selected_action)[0])
            action = self.env.actions[action_id]

            # if its not a legal action, selected another random on
            legal_actions = [a for a in self.env.get_available_moves(current) if a != "still" and "down" not in a]
            if action not in legal_actions:
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


    def random_walks_f(self, load=True):
        """
            This function does N random walks from start to goal without using any policy.
            Each walk can be max M steps long where M = 3 * length of the shortest goal.
            After X steps the walk is not random anymore and the shelter policy is used to reach the shelter.
            X = .5 * length of the shortest path to the goal
            Only walks that reach the shelter are kept
        """
        print("Random walks time")


        # Check if a file already exists:
        save_fld = "Processing/modelling/maze_path_rl/random_walks"
            
        name = "{}_rw.pkl".format(self.env.name)
        savename = os.path.join(save_fld, name)
        if not os.path.isfile(savename) or not load:

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

                # If the walk is going down (i.e. has neg Y velocity): discard it
                y_vel = -np.diff([y for x,y in walk])
                if np.any(np.where(y_vel < 0)): 
                    continue

                # If the walk isn't too long, append it to the list
                if len(walk) <= max_n_steps:
                    self.random_walks.append(walk)

            print("     {} out of {} walks were good".format(self.n_random_walks, counter))           

            # Plot each walk's X,Y, length...
            f, axarr = plt.subplots(ncols=3, nrows=2, figsize=(20, 16))
            axarr = axarr.flatten()

            # traces
            self.get_maze_image()
            axarr[0].imshow(self.maze, interpolation='none', cmap='gray')
            for w in self.random_walks:
                axarr[0].scatter([x for x,y in w], [y for x,y in w], alpha=.5)

            # x,y
            for w in self.random_walks:
                axarr[1].plot([x for x,y in w], linewidth=.75, alpha=.5)

            # Durations | number of frames per walk 
            durs = [len(w) for w in self.random_walks]
            axarr[3].hist(durs, bins=50)
            # xs, ys = [np.vstack(w)[:, 0] for w in self.random_walks], [np.vstack(w)[:, 1] for w in self.random_walks]

            # thetas = [calc_angle_between_points_of_vector((np.vstack(w)))-90 for w in self.random_walks]
            # [axarr[3].plot(t) for t in thetas]

            # Distance | euclidean distance between each two subsequent points along the walk
            dists = [np.sum(dist(np.vstack(w))) for w in self.random_walks]
            axarr[4].hist(dists, bins=50)

            # Get the distance from the direct path | X offset
            x_offsets = []
            for w in self.random_walks:
                x = [x for x,y in w]
                right, left = np.max(x), np.min(x)
                if abs(right) > abs(left):
                    x_offsets.append(right)
                else:
                    x_offsets.append(left)
            axarr[5].hist(x_offsets, bins=50)

            #  distance covered by X offset
            axarr[2].scatter(dists, x_offsets, s=20, alpha=.75)


            axarr[1].set(title="X position", xlabel="frames", ylabel="x")
            axarr[2].set(title="Paths clusters", xlabel="distance covered", ylabel="x offset")
            axarr[3].set(title="Duration (n frames)", xlabel="count", ylabel="# frames")
            axarr[4].set(title="Distance covered", xlabel="count", ylabel="tot distance")
            axarr[5].set(title="Max X offset", xlabel="count", ylabel="x offset")

            f.savefig("Processing/modelling/maze_path_rl/results/random_walks{}.png".format(self.env.name))

            # Save the random walks as a pandas dataframe
            d = dict(
                walks = self.random_walks,
                duration = durs,
                distance = dists,
                x_offset = x_offsets,
            )
            self.random_walks = pd.DataFrame.from_dict(d)

            self.random_walks.to_pickle(savename)


        else:
            self.random_walks = pd.read_pickle(savename)

    """
    -----------------------------------------------------------------------------------------------------------------
                        Finding options
    -----------------------------------------------------------------------------------------------------------------
    """
    
    def find_options(self):
        # Load the walks
        if not isinstance(self.random_walks, pd.DataFrame):
            save_fld = "Processing/modelling/maze_path_rl/random_walks"
            name = "{}_rw.pkl".format(self.env.name)
            savename = os.path.join(save_fld, name)
            self.random_walks = pd.read_pickle(savename)

        # User defined number of clusters
        n_clusters_lookup = {"PathInt":3, "PathInt2":2, "Square Maze":2, "ModelBased":3}

        # colors 
        colors = get_n_colors(3)

        # Get the two metrics we are interested in 
        X = np.array([self.random_walks['distance'].values, self.random_walks['x_offset'].values]).T


        # ! Kmeans clustering
        # # K-means clustering - fit
        # kmeans = KMeans(n_clusters=n_clusters_lookup[self.env.name])
        # kmeans.fit(X)

        # # predict
        # predictions = kmeans.predict(X)
        # centers = kmeans.cluster_centers_

        # ! other clustering
        walks = self.random_walks['walks'].values
        trajectories = [np.vstack(w) for w in walks]
        max_l = np.max([t.shape[0] for t in trajectories])
        resampled = [np.array([resample(t[:, 0], max_l), resample(t[:, 1], max_l)]) for t in trajectories]
        X = np.array(resampled)[:, 0, :]*np.array(resampled)[:, 1, :]
        cluster = AgglomerativeClustering(n_clusters=n_clusters_lookup[self.env.name], affinity='euclidean', linkage='ward')  
        predictions = cluster.fit_predict(X)

            # plot
        f, axarr = plt.subplots()
        # axarr[0].scatter(X[:, 0], X[:, 1], c=[colors[p] for p in predictions], s=100)
        # axarr[0].set(title="{} clusters".format(self.env.name), xlabel="distance travelled", ylabel="X offset")
        # axarr[0].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

        # Add predictions to dataframe
        self.random_walks['cluster'] = predictions

        # Plot each route for each cluster
        self.get_maze_image()
        axarr.imshow(self.maze, interpolation='none', cmap='gray')
        for i, row in self.random_walks.iterrows():
            axarr.scatter([x for x,y in row['walks']], [y for x,y in row['walks']],
                            s=10, alpha=.3, c=colors[row['cluster']])

        # Get the best path from each cluster (i.e. the shortest)
        kluster_walks = []
        for knum in set(predictions):
            distances = self.random_walks.loc[self.random_walks['cluster'] == knum]['distance'].values
            # x_offset = self.random_walks.loc[self.random_walks['cluster'] == knum]['x_offset'].values
            shortest = np.argmin(distances)
            k_walk = self.random_walks.loc[self.random_walks['cluster'] == knum]['walks'].values[shortest]
            axarr.scatter(line_smoother([x for x,y in k_walk]), line_smoother([y for x,y in k_walk]),
                            s=30, alpha=.8, c='r')
            kluster_walks.append(k_walk)

        self.cluster_walks = {i:w for i,w in enumerate(kluster_walks)}


        f.savefig("Processing/modelling/maze_path_rl/results/{}__options.png".format(self.env.name))



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
                        Model saving
    -----------------------------------------------------------------------------------------------------------------
    """

    def save_model(self):
        save_fld = "Processing/modelling/maze_path_rl/models"
        savename = os.path.join(save_fld, self.env.name + ".pkl")

        with open(savename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
