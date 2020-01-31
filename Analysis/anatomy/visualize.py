
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



class CellFinderScene(Scene):
    def add_cells_to_scene(self, cells, in_region=None, radius=12, **kwargs):
        if in_region is not None:
            cells = get_cells_in_region(cells, in_region)
        self.add_cells(cells, radius=radius, **kwargs)

    def add_injection_site(self, injection, wireframe=False, edit_kwargs=None, **kwargs):
        actor = self.add_from_file(injection, **kwargs)
        if wireframe:
            self.edit_actors([actor], wireframe=True)
        if edit_kwargs is not None:
            self.edit_actors([actor], **edit_kwargs)
        return actor

class CellFinderDoubleScene(DualScene):
    def __init__(self, *args, **kwargs):
        self.scenes = [CellFinderScene(*args, add_root=False, **kwargs),\
                        CellFinderScene(*args, add_root=True, **kwargs)]

    def add_cells_to_scenes(self, cells, in_region=[None, None], exclude_scene=None,  **kwargs):
        for i, (scene, region) in enumerate(zip(self.scenes, in_region)):
            if i != exclude_scene:
                if not region:
                    region = None
                scene.add_cells_to_scene(cells, in_region=region, **kwargs)

    def add_injection_sites(self, injection, exclude_scene=None, **kwargs):
        for i, scene in enumerate(self.scenes):
            if i != exclude_scene:
                actor = scene.add_injection_site(injection, **kwargs)
        return actor


if __name__ == "__main__":
    # ----------------------------- Visualize results CC mice ---------------------------- #

    scene = CellFinderDoubleScene()
    fakescene = Scene()
                    
    mice = ['CC_134_1', 'CC_134_2']
    colors = ['salmon', 'darkseagreen']

    for mouse, color in zip(mice, colors):
        ch1_cells = get_cells_for_mouse(mouse, ch=1)
        injection = get_injection_site_for_mouse(mouse, ch=1)

        # ch1_cells = ch1_cells.loc[ch1_cells.hemisphere == 'left']

        scene.add_cells_to_scenes(ch1_cells, color=color, radius=16, exclude_scene=0,
                        alpha=.6, in_region=[['SCm', 'SCs', 'IC', 'PAG'], None])

        # Add injection site and intersection
        actor = scene.add_injection_sites(injection, c=color, exclude_scene=1, alpha=.8, edit_kwargs={'smooth':True})
        actor.lighting("plastic")

        fakescene.add_brain_regions(['SCm'])
        reg = fakescene.actors['regions']['SCm']
        intersection = surfaceIntersection(actor, reg)
        scene.scenes[0].add_vtkactor(intersection)

        break


    scene.scenes[0].add_brain_regions(['SCm',], use_original_color=True, alpha=.6, wireframe=True)
    # scene.scenes[1].add_brain_regions(['MOs', 'MOp', 'RSP'], use_original_color=True, alpha=.01, wireframe=True)

    scene.render()


    # ----------------------- Visualize all injection sites ---------------------- #

    # scene = CellFinderScene(display_root=False)

    # for i, injfile in enumerate(listdir(injections_folder)):
    #     if 'ch0' in injfile: continue

    #     color = colorMap(i, vmin=0, vmax=len(listdir(injections_folder)))

    #     scene.add_injection_site(injfile, c=color, wireframe=True)

    # scene.add_brain_regions(['SCm', 'SCs'], use_original_color=True, wireframe=True)
    # scene.render()
