import sys
sys.path.append('./')

import pandas as pd
import os
from collections import namedtuple
import numpy as np
import cv2
from shutil import copyfile
from tqdm import tqdm 
from scipy import stats
import time
import random
import warnings
from shutil import copyfile
import pyexcel
import yaml


if sys.platform != "darwin":
    try: 
        dj.__version__
    except: # ? onlu import tables if we havent already 
        try:
            import datajoint as dj
            from database.dj_config import start_connection 
            dbname, _ = start_connection()    
        except:
            print("Could not connect to database")
        else:
            print("Importing tables")
            from database.TablesDefinitionsV4 import *

from Utilities.matplotlib_config import *

from Utilities.file_io.files_load_save import *
from Utilities.video_and_plotting.video_editing import Editor
from Utilities.maths.math_utils import *
from Utilities.maths.distributions import *
from Utilities.maths.filtering import *
from Utilities.constants import *

from Processing.plot.plotting_utils import close_figure, save_figure, save_all_open_figs, create_figure, show, make_legend, ortholines
from Processing.plot.plot_distributions import plot_distribution, plot_fitted_curve



