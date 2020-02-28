
import sys
sys.path.append('./')

import brainrender
brainrender.SHADER_STYLE='cartoon'

from brainrender.scene import Scene, DualScene 


import pandas as pd



from brainrender.colors import colorMap
from brainrender.Utils.videomaker import VideoMaker
from brainrender import *
from brainrender.Utils.data_manipulation import mirror_actor_at_point

import numpy as np
import os
from scipy.spatial.distance import euclidean
from skimage.filters import threshold_otsu
from vtkplotter.analysis import surfaceIntersection, extractLargestRegion

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

    scene = CellFinderScene()


    inj_folder = "Z:\\swc\\branco\\BrainSaw\\injections"
    cells_folder = "Z:\\swc\\branco\\BrainSaw\\cellfinder_cells"


    grn_mice = ['CC_136_1', 'CC_136_0']
    sc_mice = ['CC_134_1', 'CC_134_2']

    grn_colors = ['darkgreen', 'darkseagreen']
    sc_colors = ['goldenrod', 'gold']


    for mouse, color in zip(grn_mice, grn_colors):
        scene.add_injection_site(os.path.join(inj_folder, mouse+'_ch0inj.obj'), c=color)

        cells = pd.read_hdf(os.path.join(cells_folder, mouse+'_ch0_cells.h5'), key='hdf')
        scene.add_cells_to_scene(cells, color=color, radius=15, res=12, alpha=.4, 
                                in_region=['MOs', 'MOp', 'ZI', 'SCm'])


    mirror_coord = scene.get_region_CenterOfMass('root', unilateral=False)[2]
    for mouse, color in zip(sc_mice, sc_colors):
        cells = pd.read_hdf(os.path.join(cells_folder, mouse+'_ch1_cells.h5'), key='hdf')
        scene.add_cells_to_scene(cells, color=color, radius=15, res=12, alpha=.4, 
                                in_region=['MOs', 'MOp', 'ZI', 'GRN'])
        scene.add_injection_site(os.path.join(inj_folder, mouse+'_ch1inj.obj'), c=color)

        # coords = actor.points() 
        # shifted_coords = [[c[0], c[1], mirror_coord + (mirror_coord-c[2])] for c in coords]
    #     actor.points(shifted_coords)

    scene.add_brain_regions(['MOs', 'MOp', 'ZI'], use_original_color=True, alpha=.05,wireframe=False)
    scene.add_brain_regions(['GRN', 'SCm'], use_original_color=True, alpha=.5,wireframe=False)



    scene.render()


