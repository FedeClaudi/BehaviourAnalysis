import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from pandas.plotting import scatter_matrix
from collections import namedtuple
from itertools import combinations
from matplotlib import colors as mcolors



from database.NewTablesDefinitions import *
from database.dj_config import start_connection

from Processing.tracking_stats.math_utils import line_smoother
from Utilities.file_io.files_load_save import load_yaml
from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame


def cleanup_explorations(explorations):
    # Remove columns 
    to_drop = ['exploration_id', 'session_uid', 'experiment_name', 'tracking_data']
    explorations.drop(to_drop, axis=1)
    del explorations['exploration_id']

    # Remove outliers
    explorations = explorations.loc[explorations['total_travel'] < 100000]
    explorations = explorations.loc[explorations['tot_time_in_shelter'] < 750]
    explorations = explorations.loc[explorations['duration'] < 1500]

    return explorations

def expand_features(explorations):
    explorations['normalised_distance'] = np.divide(explorations['total_travel'].values, explorations['duration'].values)
    explorations['%_time_in_shelt'] = np.multiply(np.divide(explorations['tot_time_in_shelter'].values, explorations['duration'].values), 100)
    explorations['%_time_on_T'] = np.multiply(np.divide(explorations['tot_time_on_threat'].values, explorations['duration'].values), 100)
    explorations['%S/%T'] =  np.divide(explorations['%_time_in_shelt'].values, explorations['%_time_on_T'].values)

    return explorations


def plot_explorations():
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]

    explorations = pd.DataFrame(AllExplorations.fetch())
    experiments = list(set(explorations['experiment_name'].values))
    f, axarr = plt.subplots(nrows = int(len(experiments)/2), ncols=2)

    for i, row in explorations.iterrows():
        ax = axarr.flatten()[experiments.index(row['experiment_name'])]
        ax.scatter(row['tracking_data'][:, 0], row['tracking_data'][:, 1],
                    c = row['tracking_data'][:, 2], s=3, alpha=.25)

    for i, exp in enumerate(experiments):
        axarr.flatten()[i].set(title=exp)

    plt.show()


def explore_correlations_in_exploration():
    def plot_corr(df,size=10):
        """
        Function plots a graphical correlation matrix for each pair of columns in the dataframe.

        Input:
            df: pandas DataFrame
            size: vertical and horizontal size of the plot
        """

        corr = df.corr()
        fig, ax = plt.subplots(figsize=(size, size))
        ax.matshow(corr)
        plt.xticks(range(len(corr.columns)), corr.columns)
        plt.yticks(range(len(corr.columns)), corr.columns)



    explorations = pd.DataFrame(AllExplorations.fetch())
    explorations = cleanup_explorations(explorations)
    explorations = expand_features(explorations)


    scatter_matrix(explorations,  alpha=0.7, figsize=(6, 6), diagonal='kde')

    plt.show()







if __name__ == "__main__":
    print(AllExplorations())

    explore_correlations_in_exploration()


































