import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from Processing.tracking_stats.math_utils import *



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

        f, ax = plt.subplots( figsize=(8, 12))
        ax.imshow(self.maze, interpolation="none", cmap="gray")

        walks = []
        for i, _ in enumerate(env.subgoals):
            subgoal_policy = 'subgoal_{}'.format(i)
            shelter_policy = 'shelter'

            walk = []
            walk.extend(self.clean_walk(subgoal_policy, env.subgoals[i]))
            at_subgoal = len(walk)
            walk.extend(self.clean_walk(shelter_policy, env.goal, start=walk[-1]))
            distance_travelled = np.sum(calc_distance_between_points_in_a_vector_2d(np.array(walk)))
            orientation = np.abs(np.add(calc_angle_between_points_of_vector(np.array(walk[:at_subgoal])), -90))   # total orientation off the direct path until it got to subgoal
            tot_angle = np.nansum(orientation)
            walks.append((walk, distance_travelled, tot_angle, at_subgoal))

            ax.scatter([x for x,y in walk], [y for x, y in walk], c='r', alpha=.5)
            ax.scatter(env.subgoals[i][0], env.subgoals[i][1], s=60, label=str(i))

        ax.scatter(env.start[0], env.start[1], color='g', s=100)
        ax.scatter(env.goal[0], env.goal[1], color='b', s=100)
        ax.legend()

        f.savefig("Processing/modelling/maze_path_rl/results/{}_subgoals.png".format(self.learner.env.name))

        self.process_walks(walks)

        # plt.show()


    def process_walks(self, walks):
        def normalise_by_rightmedium(vector, rmidx):
            return [x/vector[rmidx] for x in vector]

        # ! walks.append((walk, distance_travelled, tot_angle, at_subgoal))


        # Get the lengths of the direct path between start and goal
        direct_vector_distance = calc_distance_between_points_2d(self.learner.env.start, self.learner.env.goal)

        sort_idxs, labels = self.path_order_lookup(self.learner.env.name)
        right_medium_idx = labels.index("Right_Medium")
        walks = [walks[i] for i in sort_idxs]

        # Get the normalised length of each path
        walk_lengths = [w[1] for w in walks]
        normalised_lengths = normalise_by_rightmedium(walk_lengths, right_medium_idx)

        # Get distance at subgoal from direct path
        position_at_subgoals = [np.array(w[0][w[3]]) for w in walks]
        direct = [np.array(self.learner.env.start), np.array(self.learner.env.goal)]
        distance_at_subgoals = [abs(calc_distane_between_point_and_line(direct, p)) for p in position_at_subgoals]
        normalised_distance_at_sg = normalise_by_rightmedium(distance_at_subgoals, right_medium_idx)

        # Get tot angle off direct path until subgoal
        angles = [w[2] for w in walks]
        normalised_angles = normalise_by_rightmedium(angles, right_medium_idx)

        # Plot
        f, axarr = plt.subplots(nrows=3, figsize=(12, 9))
        xx = np.arange(len(walk_lengths))
        colors = self.colors_lookup(labels)

        axarr[0].bar(xx, normalised_lengths, color=colors)
        axarr[0].axhline(1, linestyle="--", color="k")
        axarr[0].set(title="normalised path length", ylabel="n steps.", xticks=xx, xticklabels=labels)

        axarr[1].bar(xx, normalised_distance_at_sg, color=colors)
        axarr[1].axhline(1, linestyle="--", color="k")
        axarr[1].set(title="Normalised distance from direct path at subgoal", ylabel="px", xticks=xx, xticklabels=labels)

        axarr[2].bar(xx, normalised_angles, color=colors)
        axarr[2].axhline(1, linestyle="--", color="k")
        axarr[2].set(title="normalised Angle off direct until subgoal", ylabel="comulative theta", xticks=xx, xticklabels=labels)


        f.tight_layout()
        f.savefig("Processing/modelling/maze_path_rl/results/{}_pathsvars.png".format(self.learner.env.name))

        
    @staticmethod
    def path_order_lookup(exp):
        if exp == "PathInt":
            return [2, 1, 0], ["Left_Far", "Centre", "Right_Medium"]
        elif exp == "PathInt2":
            return [1, 0], ["Left_Far", "Right_Medium"]

    @staticmethod
    def colors_lookup(labels):
        colors = dict(
            Left_Far = [.2, .6, .2],
            Left_Medium = [.2, .6, .2],
            Centre = [.2, .2, .6],
            Right_Medium = [.6, .2, .2]
        )
        return [colors[l] for l in labels]
