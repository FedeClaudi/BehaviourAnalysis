
import sys
sys.path.append('./')

import pandas as pd
from brainrender.scene import Scene, DualScene
from brainrender import *
from vtkplotter import Plotter, interactive, Points, smoothMLS2D, recoSurface, removeOutliers, cluster, convexHull, connectedPoints, splitByConnectivity
import numpy as np
from scipy.spatial.distance import euclidean
from skimage.filters import threshold_otsu

from utils import *
from fcutils.file_io.utils import listdir

BACKGROUND_COLOR='white'
WHOLE_SCREEN=True



class CellFinderScene(Scene):
    def add_cells_to_scene(self, cells, in_region=None, radius=12, **kwargs):
        if in_region is not None:
            cells = get_cells_in_region(cells, in_region)
        self.add_cells(cells, radius=radius, **kwargs)

    def add_injection_site(self, injection, wireframe=False, **kwargs):
        actor = self.add_from_file(injection, **kwargs)
        if wireframe:
            self.edit_actors([actor], wireframe=True)

class CellFinderDoubleScene(DualScene):
    def __init__(self, *args, **kwargs):
        self.scenes = [CellFinderScene(*args, add_root=False, **kwargs),\
                        CellFinderScene(*args, add_root=True, **kwargs)]

    def add_cells_to_scenes(self, cells, in_region=[None, None], exclude_scene=None,  **kwargs):
        for i, (scene, region) in enumerate(zip(self.scenes, in_region)):
            if i != exclude_scene:
                scene.add_cells_to_scene(cells, in_region=region, **kwargs)

    def add_injection_sites(self, injection, exclude_scene=None, **kwargs):
        for i, scene in enumerate(self.scenes):
            if i != exclude_scene:
                scene.add_injection_site(injection, **kwargs)



# ----------------------------- Visualize results ---------------------------- #

scene = CellFinderDoubleScene()

                
mice = ['CC_134_1', 'CC_134_2']
colors = ['salmon', 'darkseagreen']

for mouse, color in zip(mice, colors):
    ch0_cells = get_cells_for_mouse(mouse, ch=0)
    ch1_cells = get_cells_for_mouse(mouse, ch=1)
    injection = get_injection_site_for_mouse(mouse, ch=1)

    # scene.add_cells_to_scenes(ch0_cells, color='darkseagreen', radius=16, 
    #                 exclude_scene=1, alpha=.6, in_region=[['SCm', 'SCs', 'IC', 'PAG'], ['Isocortex']])

    scene.add_cells_to_scenes(ch1_cells, color=color, radius=16,
                    alpha=.6, in_region=[['SCm', 'SCs', 'IC', 'PAG'], ['Isocortex']])

    scene.add_injection_sites(injection, c=color, exclude_scene=1, alpha=.3)


scene.scenes[0].add_brain_regions(['SCm', 'PAG', 'IC'], use_original_color=True, alpha=.2, wireframe=True)
scene.scenes[1].add_brain_regions(['MOs', 'VISp', 'AUD', 'PTLp'], use_original_color=True, alpha=.3, wireframe=True)

scene.render()

