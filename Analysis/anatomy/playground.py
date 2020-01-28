
import sys
sys.path.append('./')

import pandas as pd

from utils import get_count_by_brain_region, cellfinder_cells_folder
from fcutils.file_io.utils import listdir



for cellfile in listdir(cellfinder_cells_folder):
    cells = pd.read_hdf(cellfile, key='hdf')

    a = 1
