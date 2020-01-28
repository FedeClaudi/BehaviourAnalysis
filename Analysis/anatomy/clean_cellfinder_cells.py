import sys
sys.path.append('./')
import os
from tqdm import tqdm

import pandas as pd
from Analysis.anatomy.utils import all_regions, cellfinder_cells_folder, aba

from fcutils.file_io.utils import listdir


# Get brain region from cells
for cellfile in listdir(cellfinder_cells_folder):
    cells = pd.read_hdf(cellfile)
    print("Extracting brain region from {} [{} cells]".format(os.path.split(cellfile)[-1], len(cells)))

    regions, names = [], []
    for i, cell in tqdm(cells.iterrows()):
        p0 = list(cell[['x', 'y', 'z']].values)
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
    cells['region'] = regions
    cells['region_name'] = names
    cells.to_hdf(cellfile, key='hdf')

    
    count_per_region = cells.groupby('region').count().sort_values('type', ascending=False)['type']
    print(count_per_region[:10])
    print("\n\n")
