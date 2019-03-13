import sys
sys.path.append('./')
import matplotlib.pyplot as plt
import numpy as np
from Processing.tracking_stats.math_utils import calc_distance_between_points_in_a_vector_2d as dist
from Processing.tracking_stats.math_utils import get_n_colors, calc_angle_between_points_of_vector, calc_ang_velocity, line_smoother
from math import exp, sqrt
import json
import os
from random import choice
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.signal import resample
import random
import pickle
from collections import namedtuple


class AA:
    def __init__(self, name):
        self.models_fld = "Processing/modelling/maze_path_rl/models"
        self.results_fld = "Processing/modelling/maze_path_rl/results"
        self.name = name

        self.load_model()
        self.assign_initial_actions_value()

        # learning params
        self.n_iters = 0
        self.selected = []

        self.alpha = .25 # how much the prediction error should influence the value of each option


    def train(self):
        f, axarr = plt.subplots(ncols=3, figsize=(16, 16))
        self.visualise_maze(ax=axarr[ 0])
        # Simulate N trials and observe the total length of the path
        self.visualise_actions_values(ax=axarr[ 1])
        self.simulate_n_trials(n=1000)


        self.visualise_selected(ax=axarr[2])

        savename = os.path.join(self.results_fld, self.name+"_simulatedprobs.png")
        f.savefig(savename)



    def visualise_maze(self, ax):
        colors = get_n_colors(len(self.model.cluster_walks))
        ax.imshow(self.model.maze, interpolation='none', cmap='gray')
        for i, (w,c) in enumerate(zip(self.model.cluster_walks.values(), colors)):
            ax.scatter([x for x,y in w], [y for x,y in w], c=c, s=50, label="action {}".format(i))
        ax.legend

    def visualise_selected(self, ax):
        colors = get_n_colors(len(self.model.cluster_walks))
        sels = [self.selected.count(s)/self.n_iters for s in sorted(set(self.selected))]
        x = np.arange(len(colors))

        ax.bar(x, sels, color=colors)


    def load_model(self):
        load_name = os.path.join(self.models_fld, self.name+".pkl")
        with open(load_name, 'rb') as inp:
            self.model = pickle.load(inp)


    def assign_initial_actions_value(self):
        # Get the relative length of each action
        actions = self.model.cluster_walks.values()
        action_lengths = [len(a) for a in actions]
        actions_sdev = [sqrt(a) for a in action_lengths]
        a = namedtuple("action", "mu sd")
        self.actions = {i:a(mu, sdev) for i, (mu, sdev) in enumerate(zip(action_lengths, actions_sdev))}

        self.observed_outcomes = {i:[] for i in self.actions.keys()}


    def visualise_actions_values(self, ax=None):
        if ax is None: f, ax = plt.subplots()
        colors = get_n_colors(len(self.actions))


        for i, (mu, omega) in self.actions.items():
            action_samples = np.random.normal(loc=mu, scale=omega, size=10000)
            ax.hist(action_samples, bins=100, color=colors[i], alpha=.75, label="action: {}".format(i))
        ax.legend()
        ax.set(title = self.name + " iters: " + str(self.n_iters))



    def simulate_n_trials(self, n=10):
        a = namedtuple("action", "mu sd")

        for i in np.arange(n):
            print(" trial: ", i)
            self.n_iters += 1

            # Sample a value for each action
            sampled_action_values = [np.random.normal(m, o) for m,o in self.actions.values()]

            # Selected  best valued action
            selected_action = np.argmin(sampled_action_values)

            # Get the prediction error
            obs_value = sampled_action_values[selected_action]
            self.observed_outcomes[selected_action].append(obs_value)

            exp_val, sd = self.actions[selected_action]
            prediction_error = -(exp_val - obs_value)


            # update the action value
            updated_value = exp_val + self.alpha * prediction_error
            # self.actions[selected_action] = a(updated_value, sd)

            self.selected.append(selected_action)

