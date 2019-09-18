# %%
import sys
sys.path.append('./') 
from anatomy.connectivity_analyzer import ConnectivityAnalyzer
import os
analyzer = ConnectivityAnalyzer()

#%%
from vtkplotter import *

# embedWindow('itkwidgets')

#%%
sets = analyzer.other_sets
sets_names = sorted(list(sets.keys()))

#%%
hypothalamus_structures = list(sets["Summary structures of the hypothalamus"].acronym.values)
thalamus_structures = list(sets["Summary structures of the thalamus"].acronym.values)
pons_structures = list(sets["Summary structures of the pons"].acronym.values)
midbrain_structures = list(sets["Summary structures of the midbrain"].acronym.values)


#%%
neurons_fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\anatomy\\Mouse Light"
neurons_file = os.path.join(neurons_fld, "one_neuron.json")

# %%
# analyzer.plot_structures_3d(["PAG", "SCm", "ZI", "GRN", "PPN", "SCs"], verbose=True, sagittal_slice=False,
#                                 default_colors=True,
#                                 target = "CUN",
#                                 target_color="red",
#                                 others_color="palegoldenrod",
#                                 others_alpha=1,
#                                 neurons_file = None,
#                                 specials=["PAG", "SCm", "ZI"],
#                                 notebook=False)
analyzer.plot_structures_3d(["ZI"], verbose=True, sagittal_slice=False,
                                default_colors=True,
                                target = "LHA",
                                target_color="red",
                                others_color="palegoldenrod",
                                others_alpha=1,
                                neurons_file = None,
                                specials=["PAG", "SCm", "ZI"],
                                notebook=False)

#%%
# videopath = os.path.join(analyzer.main_fld, "Videos", "oneneuron2.mp4")
# analyzer.video_maker(videopath, ["PAG", "SCm", "ZI", "GRN"], verbose=True, sagittal_slice=False,
#                                 default_colors=True,
#                                 target = None,
#                                 target_color="red",
#                                 others_color="palegoldenrod",
#                                 others_alpha=1,
#                                 neurons_file = neurons_file,
#                                 specials=["PAG", "SCm", "ZI"],
#                                 notebook=False)