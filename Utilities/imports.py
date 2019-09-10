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
from math import sqrt, factorial


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

from Utilities.matplotlib_config_figures import *

from Utilities.file_io.files_load_save import *
from Utilities.video_and_plotting.video_editing import Editor
from Utilities.maths.math_utils import *
from Utilities.maths.distributions import *
from Utilities.maths.filtering import *
from Utilities.constants import *

from Processing.plot.plotting_utils import *
from Processing.plot.plot_distributions import *



sns.set_style("white", {
            "axes.grid":"False",
            "ytick.right":"False",
            "ytick.left":"True",
            "xtick.bottom":"True",
            "text.color": "0"
})
mpl.rc('text', usetex=False)

params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 150,  # to adjust notebook inline plot size
    'axes.labelsize': 8, # fontsize for x and y labels (was 10)
    'axes.titlesize': 8,
    'font.size': 8, # was 10
    'legend.fontsize': 6, # was 10
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'text.usetex': False,        # ! <----- use TEX
    'figure.figsize': [3.39, 2.10],
}
