# %%
import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project
from Utilities.imports import *
get_ipython().magic('matplotlib inline')

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\anatomy\\mouse_connectivity"
manifest = os.path.join(fld, "manifest.json")

mcc = MouseConnectivityCache(manifest_file=manifest)

# %% 
# ? Get brain structures
# grab the StructureTree instance
structure_tree = mcc.get_structure_tree()

# get info on some structures
structures = structure_tree.get_structures_by_name(["Periaqueductal gray", 'Superior colliculus, motor related'])
pd.DataFrame(structures)

SC = structure_tree.get_structures_by_name(['Superior colliculus, motor related'])[0]
PAG = structure_tree.get_structures_by_name(["Periaqueductal gray"])[0]

# %%
# ? get brainz templates
template, template_info = mcc.get_template_volume()
annot, annot_info = mcc.get_annotation_volume()

# in addition to the annotation volume, you can get binary masks for individual structures
# in this case, we'll get one for the SC
sc_mask, cm_info = mcc.get_structure_mask(294)

print("# slices in template: ", template.shape[0])

# %%
# ? Plot coronal slices
ncol, nrow= 5, 5
f, axarr = create_figure(subplots=True, ncols=ncol, nrows=nrow)

for i, sl in enumerate(np.linspace(0, template.shape[2]-1, ncol*nrow)):
    # axarr[i].imshow(template[np.int(sl),:,:], origin="upper", cmap='gray', aspect='equal', vmin=template.min(), vmax=template.max())
    axarr[i].imshow(template[:, :, np.int(sl)], origin="upper", cmap='gray', aspect='equal', vmin=template.min(), vmax=template.max())






#%%
# ? projection matrix

# Get WT injections into SC
sc = structure_tree.get_structures_by_acronym(['SCm'])[0]
sc_experiments = mcc.get_experiments(cre=False, 
                                       injection_structure_ids=[sc['id']])

print("%d SC experiments" % len(sc_experiments))

structure_unionizes = mcc.get_structure_unionizes([ e['id'] for e in sc_experiments ], 
                                                  is_injection=False,
                                                  structure_ids=[SC['id']],
                                                  include_descendants=True)

print("%d SC non-injection, SC structure unionizes" % len(structure_unionizes))




#%%
sc_experiment_ids = [ e['id'] for e in sc_experiments ]
targets = structure_tree.child_ids([SC['id'], PAG["id"]])[0]  # Get structures you want to look at 

pm = mcc.get_projection_matrix(experiment_ids = sc_experiment_ids, 
                               projection_structure_ids = targets,
                               hemisphere_ids= [2], # right hemisphere, ipsilateral
                               parameter = 'projection_density')

row_labels = pm['rows'] # these are just experiment ids
column_labels = [ c['label'] for c in pm['columns'] ] 
matrix = pm['matrix']

fig, ax = plt.subplots(figsize=(15,15))
heatmap = ax.pcolor(matrix, cmap=plt.cm.afmhot)

# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(matrix.shape[1])+0.5, minor=False)
ax.set_yticks(np.arange(matrix.shape[0])+0.5, minor=False)

ax.set_xlim([0, matrix.shape[1]])
ax.set_ylim([0, matrix.shape[0]])          

# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()

ax.set_xticklabels(column_labels, minor=False)
ax.set_yticklabels(row_labels, minor=False)
plt.show()



#%%
# ? look at SC experiments grid data

for experiment_id in sc_experiment_ids:
    # projection density: number of projecting pixels / voxel volume
    pd, pd_info = mcc.get_projection_density(experiment_id)

    # injection density: number of projecting pixels in injection site / voxel volume
    ind, ind_info = mcc.get_injection_density(experiment_id)

    # injection fraction: number of pixels in injection site / voxel volume
    inf, inf_info = mcc.get_injection_fraction(experiment_id)

    # data mask:
    # binary mask indicating which voxels contain valid data
    dm, dm_info = mcc.get_data_mask(experiment_id)
#%%
# compute the maximum intensity projection (along the anterior-posterior axis) of the projection data
pd_mip = pd.max(axis=0)
ind_mip = ind.max(axis=0)
inf_mip = inf.max(axis=0)

# show that slice of all volumes side-by-side
f, pr_axes = plt.subplots(1, 3, figsize=(15, 6))

pr_axes[0].imshow(pd_mip, cmap='hot', aspect='equal')
pr_axes[0].set_title("projection density MaxIP")

pr_axes[1].imshow(ind_mip, cmap='hot', aspect='equal')
pr_axes[1].set_title("injection density MaxIP")

pr_axes[2].imshow(inf_mip, cmap='hot', aspect='equal')
pr_axes[2].set_title("injection fraction MaxIP")

plt.show()


#%%
