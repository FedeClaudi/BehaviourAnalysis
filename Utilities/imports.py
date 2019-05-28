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

if sys.platform == "darwin":
    import matplotlib
    matplotlib.use("Qt5Agg")
    
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
        import datajoint as dj
        from database.dj_config import start_connection 
        dbname, _ = start_connection()    
        from database.TablesDefinitionsV4 import *

from Utilities.file_io.files_load_save import *
from Utilities.video_and_plotting.video_editing import Editor

from Utilities.maths.math_utils import *


