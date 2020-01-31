import sys
sys.path.append('./')

import pandas as pd
from brainrender.scene import Scene, DualScene
from brainrender.colors import colorMap
from brainrender import *
import numpy as np
from scipy.spatial.distance import euclidean
from skimage.filters import threshold_otsu
from vtkplotter.analysis import surfaceIntersection

from utils import *
from fcutils.file_io.utils import listdir

BACKGROUND_COLOR='white'
WHOLE_SCREEN=True


mouse = 'CC_134_1'
color = 'salmon'

camera = dict(
    position = [1379.055, -3165.463, 28921.812] ,
    focal = [6919.886, 3849.085, 5688.164],
    viewup = [0.171, -0.954, -0.247],
    distance = 24893.917,
    clipping = [9826.898, 43920.235] ,
)

def set_camera(scene):
    scene.rotated = True
    scene.plotter.camera.SetPosition(camera['position'])
    scene.plotter.camera.SetFocalPoint(camera['focal'])
    scene.plotter.camera.SetViewUp(camera['viewup'])
    scene.plotter.camera.SetDistance(camera['distance'])
    scene.plotter.camera.SetClippingRange(camera['clipping'])

# # -------------------------------- CELLS SCENE ------------------------------- #

# scene = Scene()
                    

# ch1_cells = get_cells_in_region(get_cells_for_mouse(mouse, ch=1), ['Isocortex'])

# scene.add_cells(ch1_cells, color=color, radius=16, alpha=.6)
# set_camera(scene)
# scene.render()

# ------------------------------- INJSITE SCENE ------------------------------ #

scene = Scene()

injection = get_injection_site_for_mouse(mouse, ch=1)

actor = scene.add_from_file(injection, ) 
actor.color(color).alpha(.9)
actor.lighting("plastic")

scene.add_brain_regions(['SCm'], use_original_color=True, alpha=.7, wireframe=True)
reg = scene.actors['regions']['SCm']
intersection = surfaceIntersection(actor, reg)
scene.add_vtkactor(intersection)

set_camera(scene)
scene.render()

