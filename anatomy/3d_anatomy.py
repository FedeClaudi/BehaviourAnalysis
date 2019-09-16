# %%
import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import namedtuple
import vtk

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.api.queries.ontologies_api import OntologiesApi
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi




#%%
fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\anatomy\\mouse_connectivity"
save_fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\anatomy\\plots\\3d"
manifest = os.path.join(fld, "manifest.json")
mcc = MouseConnectivityCache(manifest_file=manifest)

structure_tree = mcc.get_structure_tree()


space = ReferenceSpaceApi()

#%%
structure = structure_tree.get_structures_by_acronym(["PAG"])
mesh = space.download_structure_mesh(structure_id = structure[0]["id"], ccf_version ="annotation/ccf_2017", file_name=os.path.join(save_fld, "test.obj"))

#%%
fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\anatomy"
save_fld = os.path.join(fld, "3dModels")
structures = ['root', "PAG", "SCm"]

for structure in structures:
    struct = structure_tree.get_structures_by_acronym([structure])
    mesh = space.download_structure_mesh(structure_id = struct[0]["id"], ccf_version ="annotation/ccf_2017", file_name=os.path.join(save_fld, "{}.obj".format(structure)))



#%%
vp = Plotter(title='first example')


colors = [[.6, .6, .6], [.8, .4, .4], [.4, .4, .8]]
alphas = [.3, 1, 1]
for structure, color, alpha in zip(structures, colors, alphas):
    obj_path = os.path.join(save_fld, "{}.obj".format(structure))
    vp.load(obj_path, c=color, alpha=alpha) 
vp.show()  #



#%%
