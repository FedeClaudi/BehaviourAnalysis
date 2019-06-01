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

import matplotlib as mpl
if sys.platform == "darwin":
    mpl.use("Qt5Agg")

# Set up matplotlib
mpl.rcParams['text.color'] = "white"

mpl.rcParams['figure.figsize'] = [20, 16]
mpl.rcParams['figure.facecolor'] = [.1, .1, .1]
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['savefig.facecolor'] = [.1, .1, .1]

mpl.rcParams['figure.subplot.left'] = .05
mpl.rcParams['figure.subplot.right'] = .95
mpl.rcParams['figure.subplot.bottom'] = .1
mpl.rcParams['figure.subplot.top'] = .95
mpl.rcParams['figure.subplot.wspace'] = .1
mpl.rcParams['figure.subplot.hspace'] = .1



mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'

mpl.rcParams['lines.linewidth'] = 2.0

mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.loc'] = 'best'
mpl.rcParams['legend.numpoints'] = 2
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['legend.framealpha'] = .8
mpl.rcParams['legend.scatterpoints'] = 3
mpl.rcParams['legend.edgecolor'] = 'red'
mpl.rcParams['legend.facecolor'] = [.2, .2, .2]


mpl.rcParams['axes.facecolor'] = [.4, .4, .4]
mpl.rcParams['axes.edgecolor'] = "white"
mpl.rcParams['axes.labelcolor'] = "white"

mpl.rcParams['text.color'] = "white"


mpl.rcParams['image.aspect'] = "auto"


import matplotlib.pyplot as plt
import seaborn as sns

import time
import random
import warnings
from shutil import copyfile
import pyexcel
import yaml

if sys.platform != "darwin":
    try:
        dj.__version__
    except:
        try:
            import datajoint as dj
            from database.dj_config import start_connection 
            dbname, _ = start_connection()    
            from database.TablesDefinitionsV4 import *
        except:
            pass

from Utilities.file_io.files_load_save import *
from Utilities.video_and_plotting.video_editing import Editor
from Utilities.maths.math_utils import *




