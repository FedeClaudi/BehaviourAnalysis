
import sys
sys.path.append('./')

import pandas as pd
from brainrender.scene import Scene, DualScene
from brainrender import *
from vtkplotter import Plotter, interactive

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

    def add_cells_to_scenes(self, cells, in_region=[None, None], **kwargs):
        for scene, region in zip(self.scenes, in_region):
            scene.add_cells_to_scene(cells, in_region=region, **kwargs)

    def add_injection_sites(self, injection, **kwargs):
        for scene in self.scenes:
            scene.add_injection_site(injection, **kwargs)

mouse = "CC_134_1"

cells = get_cells_for_mouse(mouse, ch=1)
injection = get_injection_site_for_mouse(mouse, ch=1)


scene = CellFinderDoubleScene()

scene.add_cells_to_scenes(cells, color='salmon', radius=18, in_region=[['SCm', 'PAG'], ['SCm', 'PAG', 'Isocortex']])
scene.add_injection_sites(injection, c='green', alpha=1)

scene.scenes[0].add_brain_regions(['SCm', 'PAG'], use_original_color=True, alpha=.3, wireframe=True)
scene.scenes[1].add_brain_regions(['SCm', 'PAG', 'MOs', 'VISp', 'AUD'], use_original_color=True, alpha=.3, wireframe=True)

scene.render()