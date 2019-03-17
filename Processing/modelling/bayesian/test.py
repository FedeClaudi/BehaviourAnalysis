import numpy as np
import scipy.stats as stats
import pymc3 as pm
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import theano.tensor as tt

from database.NewTablesDefinitions import *
from database.database_fetch import *

from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame, get_arm_given_rois, convert_roi_id_to_tag
from Processing.tracking_stats.math_utils import get_roi_enters_exits, line_smoother, calc_distance_between_points_2d, remove_tracking_errors


# Get data [L-R escapes]
asym_exps = ["PathInt2", "PathInt2-L"]
sym_exps = ["Square Maze", "TwoAndahalf Maze"]

asym = [arm for arms in [get_trials_by_exp(e, 'true', ['escape_arm']) for e in asym_exps] for arm in arms]
sym = [arm for arms in [get_trials_by_exp(e, 'true', ['escape_arm']) for e in sym_exps] for arm in arms]

asym_escapes = np.array([1 if 'Right' in e else 0 for e in asym])
sym_escapes = np.array([1 if 'Right' in e else 0 for e in sym])


np.save('Processing/modelling/bayesian/asym.npy', np.array)
np.save('Processing/modelling/bayesian/asym.npy', np.array)




