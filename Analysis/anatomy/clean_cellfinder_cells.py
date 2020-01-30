import sys
sys.path.append('./')
import os
from tqdm import tqdm
import pandas as pd

from Analysis.anatomy.utils import all_regions, cellfinder_cells_folder, aba

from fcutils.file_io.utils import listdir
from brainrender.scene import Scene

# Get center of mass of the whole brain to determine hemisphere of each cell
scene = Scene()
root_com = scene.get_region_CenterOfMass('root')

# Get brain region from cells
for cellfile in listdir(cellfinder_cells_folder):
    try:
        cells = pd.read_hdf(cellfile)
    except:
        cells = pd.read_hdf(cellfile, key='hdf')
    print("Extracting brain region from {} [{} cells]".format(os.path.split(cellfile)[-1], len(cells)))

    regions, names, hemispheres = [], [], []
    for i, cell in tqdm(cells.iterrows()):
        p0 = list(cell[['x', 'y', 'z']].values)

        # Get the brain region the cell belongs to
        region = aba.get_structure_from_coordinates(p0)
        if region is None:
            regions.append(None)
            names.append(None)
        else:
            acro = None
            for region_name, acronyms in all_regions.items():
                if region['acronym'] in acronyms:
                    acro = region_name
                    break
            
            if acro is None:
                acro = region['acronym']

            regions.append(acro)
            names.append(region['name'])

        # Get the hemisphere the cell belongs to
        if p0[2] > root_com[2]:
            hemispheres.append('right')
        else:
            hemispheres.append('left')

    cells['region'] = regions
    cells['region_name'] = names
    cells['hemisphere'] = hemispheres
    cells.to_hdf(cellfile, key='hdf')

    
    count_per_region = cells.groupby('region').count().sort_values('type', ascending=False)['type']
    print(count_per_region[:10])
    print("\n\n")
