# %%
import sys
sys.path.append('./')  
from anatomy.connectivity_analyzer_v2 import ConnectivityAnalyzer


# %%
analyzer = ConnectivityAnalyzer()
#
# %%
SOI = "PAG"

if not SOI in analyzer.structures.acronym.values: 
    raise ValueError("{} is not a valid structure! {}".format(SOI, sorted(analyzer.structures.acronym.values)))
    

# %%
efferents = analyzer.analyse_efferents(SOI, projection_metric="normalized_projection_volume")
efferents


# %%
afferents = analyzer.analyze_afferents(SOI, projection_metric="normalized_projection_volume")
afferents

#%%
print(["{}".format(x) for x in afferents.origin_acronym.values[-10:]])

#%%
