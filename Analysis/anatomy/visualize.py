
import sys
sys.path.append('./')

import pandas as pd
from brainrender.scene import Scene

from utils import get_count_by_brain_region, cellfinder_cells_folder, get_cells_for_mouse, get_injection_site_for_mouse
from fcutils.file_io.utils import listdir


mouse = "CC_134_1"

cells = get_cells_for_mouse(mouse, ch=0)
injection = get_injection_site_for_mouse(mouse, ch=0)



scene = Scene()

scene.add_cells(cells, color='darkseagreen', radius=15)
scene.add_from_file(injection, c='gree')

scene.add_brain_regions(['SCm'], use_original_color=True, alpha=.6, wireframe=True)

scene.render()