
import sys
sys.path.append('./')

import numpy as np
import pandas as pd
import scipy.stats as stats
import math
import matplotlib.mlab as mlab
import matplotlib as mpl
import matplotlib.pyplot as plt

from Processing.choice_analysis.chioces_visualiser import ChoicesVisualiser as chioce_data
from database.NewTablesDefinitions import *

from database.database_fetch import *


# Manually plot data from the closing arm two arms asym stuff

mice = [
    ([1, 1, 0, 0, 1, 1, 0, 1, 0], 2),
    ([1, 1, 1, 1, 1, 1, 1], 3),
    ([1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1], 2),
    ([1, 1, 1, 1, 0, 1, 0], 3),
    ([1, 1, 1, 1, 1, 1, 0, 1, 0], 3),
    ([1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0], 4),
    ([1, 1, 1, 1, 0, 1, 0], 3),



]


