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



def plot_explorations():
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]

    data = pd.DataFrame(AllExplorations().fetch())
    experiments = set(list(data['experiment_name'].values))

    f, axarr = plt.subplots(len(experiments), 1)
    for exp_i, exp in enumerate(experiments):
        print(exp)
        ax = axarr[exp_i]
        exp_data = data.loc[data['experiment_name']==exp]
        for index, row in exp_data.iterrows():
            print(index)
            tracking_data = row['tracking_data']
            x,y = tracking_data[:, 0], tracking_data[:, 1]
            rois_ids = [int(r) for  r in set(tracking_data[:, -1])]

            for rid in rois_ids:          
                select = np.where(tracking_data[:, -1]==rid)[0]
                ax.scatter(x[select], y[select], c=sorted_names[rid*5], alpha=.3)

    plt.show()







if __name__ == "__main__":
    print(AllExplorations())

    plot_explorations()


































