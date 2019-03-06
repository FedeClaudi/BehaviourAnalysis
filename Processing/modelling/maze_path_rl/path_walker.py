import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from Processing.tracking_stats.math_utils import calc_distance_between_points_in_a_vector_2d, calc_angle_between_points_of_vector



class Walker:
    def __init__(self, learner):
        self.learner = learner
        self.initial_position()

        # params
        self.direct_vector_steps = 3  # steps in which shelter distance policy dominates
        self.distance_steps = 10      # steps in which start distance policy dominates
        self.direct_vector_p = .8     # weight to give to policy while it dominates
        self.distance_p = .3  
    
        # other params
        self.n_walks_per_start = 3
        self.max_steps = 100


    def initial_position(self):
        start = self.learner.env.start
        offsets = [-3, -2, -1, 0, 1, 2, 3]

        starts = []
        for offset in offsets:
            s = start.copy()
            s[0] +=  offset
            starts.append(s)

        self.starts = starts


    def walk(self):
        self.walks = []
        for start in self.starts: # loop over each starting position
            start_walks = []

            for i in np.arange(self.n_walks_per_start):  # do N walks per starting location
                walk = []
                # curr = self.learner.env.start.copy()
                curr = start.copy()
                step_n = 0

                # do each step
                while curr != self.learner.env.goal and step_n < self.max_steps:
                    step_n += 1
                    walk.append(curr.copy())

                    # randomly select steps threshold
                    dv_steps = np.random.normal(self.direct_vector_p, self.direct_vector_p/2, 1)
                    aw_steps = np.random.normal(self.distance_steps, self.distance_steps/3, 1)

                    # select policy
                    if step_n <= dv_steps:
                        p = self.direct_vector_p
                        policy = "direct_vector"
                    elif step_n <= aw_steps:
                        p = self.distance_steps
                        policy = "away"
                    else:
                        p = 0
                        policy = "direct_vector"

                    # calculate two alternative actions
                    shelter_next = self.step("direct_vector", curr)
                    policy_next = self.step(policy, curr)

                    # select one
                    select = np.random.uniform(0, 1, 1)
                    if policy != "shelter" and select <= p:
                        nxt = policy_next
                    else:
                        nxt = shelter_next

                    # Check that nxt is a legal move
                    if nxt[0] < 0 or nxt[1]< 0 or nxt[0] > self.learner.env.grid_size or nxt[1]>self.learner.env.grid_size: break
                    if nxt in self.learner.env.free_states: curr = nxt

                # Append to stuff
                start_walks.append(walk)
            self.walks.append(start_walks)

    def clean_walk(self, policy, goal, start=None):
        walk = []
        if start is None:
            curr = self.learner.env.start.copy()
        else:
            curr = start.copy()

        step_n = 0

        # do each step
        while curr != goal and step_n < self.max_steps:
            step_n += 1
            walk.append(curr.copy())

            nxt = self.step(policy, curr)

            # Check that nxt is a legal move
            if nxt[0] < 0 or nxt[1] < 0 or nxt[0] > self.learner.env.grid_size or nxt[1] > self.learner.env.grid_size:
                break
            if nxt in self.learner.env.free_states:
                curr = nxt
        walk.append(curr)
        
        return walk



    def step(self, policy, current):
        nxt = current.copy()
        action_values = self.learner.Q[policy][current[0]][current[1]]
        action = np.argmax(action_values)
        action = self.learner.env.actions[action]

        if action == "down":
            nxt[1] -= 1
        elif action == "right":
            nxt[0] += 1
        elif action == "left":
            nxt[0] -= 1
        elif action == "up":
            nxt[1] += 1
        elif action == "up-right":
            nxt[0] += 1
            nxt[1] += 1
        elif action == "up-left":
            nxt[0] -= 1
            nxt[1] += 1
        elif action == "down-right":
            nxt[0] += 1
            nxt[1] -= 1
        elif action == "down-left":
            nxt[0] -= 1
            nxt[1] -= 1

        return nxt


    def policy_plot(self):
        model = self.learner

        Q = model.Q
        start = model.env.start
        free = model.env.free_states

        
        grid_size = model.env.grid_size
        name = model.env.name

        f, axarr = plt.subplots(ncols=3, nrows=len(list(model.Q.keys())), figsize=(16, 12))

        for i, (Q_name, Q) in enumerate(model.Q.items()):
            if not "subgoal" in Q_name:
                goal = model.env.goal
            else:
                n = int(Q_name.split("_")[-1])
                goal = model.env.subgoals[n]


            grid_size = len(Q)
            pol = [[max(Q[i][j]) for i in range(grid_size)] for j in range(grid_size)]
            maze = [[1 if [i, j] in free else 0 for i in range(
                grid_size)] for j in range(grid_size)]
            act = [[np.argmax(Q[i][j]) if pol[j][i] > 0 else np.nan for i in range(
                grid_size)] for j in range(grid_size)]


            axarr[i, 0].imshow(maze, interpolation='none', cmap='gray')
            axarr[i, 0].plot(start[0], start[1], 'o', color='g')
            axarr[i, 0].plot(goal[0], goal[1], 'o', color='b')

            walk = self.clean_walk(Q_name, self.learner.env.goal)
            x, y = [x for x, y in walk], [y for x, y in walk]
            axarr[i, 0].scatter(x, y, c='r', alpha=0.5, s=10)
            axarr[i, 0].plot(x[0], y[0], 'o', color='g')


            axarr[i, 1].imshow(act, interpolation='none',  cmap="tab20")
            axarr[i, 2].imshow(pol, interpolation='none')

            axarr[i, 0].set(title=list(model.Q.keys())[i])
            axarr[i, 1].set(title="actions")
            axarr[i, 2].set(title="Policy")

        axarr = axarr.flatten()
        for ax in axarr:
            ax.set(xticks=[], yticks=[])
        f.tight_layout()
        f.savefig("Processing/modelling/maze_path_rl/results/{}.png".format(name))

        self.maze = maze

    def walks_plot(self):
        f, ax = plt.subplots()
        ax.imshow(self.maze, interpolation="none", cmap="gray")
        for walks in self.walks:
            for walk in walks:
                x, y = [x for x,y in walk], [y for x,y in walk]
                ax.scatter(x,y, alpha=.4)

        
        f.savefig("Processing/modelling/maze_path_rl/results/{}_walks.png".format(self.learner.env.name))


    def subgoals_plot(self):
        env = self.learner.env

        f, axarr = plt.subplots(nrows=3, figsize=(8, 12))
        ax = axarr[0]
        ax.imshow(self.maze, interpolation="none", cmap="gray")

        walks = []
        for i, _ in enumerate(env.subgoals):
            subgoal_policy = 'subgoal_{}'.format(i)
            shelter_policy = 'shelter'

            walk = []
            walk.extend(self.clean_walk(subgoal_policy, env.subgoals[i]))
            walk.extend(self.clean_walk(shelter_policy, env.goal, start=walk[-1]))
            distance_travelled = np.sum(calc_distance_between_points_in_a_vector_2d(np.array(walk)))
            orientation = np.abs(np.add(calc_angle_between_points_of_vector(np.array(walk)), -90))
            tot_angle = np.nansum(orientation)
            walks.append((walk, distance_travelled, tot_angle))

            ax.scatter([x for x,y in walk], [y for x, y in walk], c='r', alpha=.5)
            ax.scatter(env.subgoals[i][0], env.subgoals[i][1], s=60, label=str(i) + "__" + str(round(distance_travelled)) + "__"+str(tot_angle))

        ax.scatter(env.start[0], env.start[1], color='g', s=100)
        ax.scatter(env.goal[0], env.goal[1], color='b', s=100)
        ax.legend()

        walks_lengts = [l for w,l,a in walks]
        min_l = min(walks_lengts)
        normalised_lengths = [abs(1 - l/min_l) for l in walks_lengts]

        walks_angles = [a for w,l,a in walks]
        min_a = min(walks_angles)
        normalised_angles = [abs(1  - a/min_a) for a in walks_angles]

        x = np.arange(len(walks))

        axarr[1].bar(x, normalised_lengths)
        axarr[1].set(title="normalised paths lenghts", xticks=x, xticklabels=[str(xx) for xx in x])
        axarr[2].bar(x, normalised_angles)
        axarr[2].set(title="normalised paths angle", xticks=x, xticklabels=[str(xx) for xx in x])



        f.savefig("Processing/modelling/maze_path_rl/results/{}_subgoals.png".format(self.learner.env.name))

        # plt.show()

