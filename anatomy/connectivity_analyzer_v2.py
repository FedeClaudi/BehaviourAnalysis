import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import namedtuple
from vtkplotter import Plotter

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.api.queries.ontologies_api import OntologiesApi
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi


class ConnectivityAnalyzer:
    fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\anatomy\\mouse_connectivity"
    models_fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\anatomy\\3dModels"

    hemispheres = namedtuple("hemispheres", "left right both") # left: CONTRA, right: IPSI
    hemispheres_names = ["left", "right", "both"]

    density_metric = "projection_energy"

    def __init__(self):
        manifest = os.path.join(self.fld, "manifest.json")
        self.mcc = MouseConnectivityCache(manifest_file=manifest)

        self.structure_tree = self.mcc.get_structure_tree()
        
        # Get the main structures names, ID...
        oapi = OntologiesApi()
        summary_structures = self.structure_tree.get_structures_by_set_id([167587189])
        self.structures = pd.DataFrame(summary_structures)

        # get reference space
        self.space = ReferenceSpaceApi()

        # save folder for experiments pickle data
        self.save_fld = os.path.join(self.fld, "fc_experiments_unionized")



    def load_all_experiments(self):
        # Downloads all experiments from allen brain atlas and saves the results as an easy to read pkl file
        for acronym in self.structures.acronym.values:
            print("Fetching experiments for : {}".format(acronym))

            structure = self.structure_tree.get_structures_by_acronym([acronym])[0]
            experiments = self.mcc.get_experiments(cre=False, injection_structure_ids=[structure['id']])

            print("     found {} experiments".format(len(experiments)))

            try:
                structure_unionizes = self.mcc.get_structure_unionizes([e['id'] for e in experiments], 
                                                            is_injection=False,
                                                            structure_ids=self.structures.id.values,
                                                            include_descendants=False)
            except: pass
            structure_unionizes.to_pickle(os.path.join(self.save_fld, "{}.pkl".format(acronym)))


    def analyse_efferents(self, SOI, projection_metric = None):
        """[Loads the experiments on SOI and looks at average statistics of efferent projections]
        
        Arguments:
            SOI {[str]} -- [acronym of the structure of interest to look at]
        """
        if projection_metric is None: 
            projection_metric = self.projection_metric

        experiment_data = pd.read_pickle(os.path.join(self.save_fld, "{}.pkl".format(SOI)))

        # Loop over all structures and get the injection density
        results = {"left":[], "right":[], "both":[], "target_id":[], "target_acronym":[], "target_name":[]}
        for target in self.structures.id.values:
            target_acronym = self.structures.loc[self.structures.id == target].acronym.values[0]
            target_name = self.structures.loc[self.structures.id == target].name.values[0]

            exp_target = experiment_data.loc[experiment_data.structure_id == target]

            exp_target_hemi = self.hemispheres(exp_target.loc[exp_target.hemisphere_id == 1], exp_target.loc[exp_target.hemisphere_id == 2], exp_target.loc[exp_target.hemisphere_id == 3])
            proj_energy = self.hemispheres(np.nanmean(exp_target_hemi.left[projection_metric].values),
                                            np.nanmean(exp_target_hemi.right[projection_metric].values),
                                            np.nanmean(exp_target_hemi.both[projection_metric].values)
            )

            # print("{} to {} -> L:{}, R:{}, B:{}".format(SOI, target_acronym, round(proj_energy.left, 4), round(proj_energy.right, 4), round(proj_energy.both, 4)))
            for hemi in self.hemispheres_names:
                results[hemi].append(proj_energy._asdict()[hemi])
            results["target_id"].append(target)
            results["target_acronym"].append(target_acronym)
            results["target_name"].append(target_name)

        results = pd.DataFrame.from_dict(results).sort_values("right")
        return results


    def analyze_afferents(self, SOI, projection_metric = None):
        """[look at all areas projecting to it]
        
        Arguments:
            SOI {[str]} -- [structure of intereset]
        """
        if projection_metric is None: 
            projection_metric = self.projection_metric
        SOI_id = self.structure_tree.get_structures_by_acronym([SOI])[0]["id"]

        # Loop over all strctures and get projection towards SOI
        results = {"left":[], "right":[], "both":[], "origin_id":[], "origin_acronym":[], "origin_name":[]}

        for origin in self.structures.id.values:
            origin_acronym = self.structures.loc[self.structures.id == origin].acronym.values[0]
            origin_name = self.structures.loc[self.structures.id == origin].name.values[0]

            experiment_data = pd.read_pickle(os.path.join(self.save_fld, "{}.pkl".format(origin_acronym)))

            exp_target = experiment_data.loc[experiment_data.structure_id == SOI_id]
            exp_target_hemi = self.hemispheres(exp_target.loc[exp_target.hemisphere_id == 1], exp_target.loc[exp_target.hemisphere_id == 2], exp_target.loc[exp_target.hemisphere_id == 3])
            proj_energy = self.hemispheres(np.nanmean(exp_target_hemi.left[projection_metric].values),
                                            np.nanmean(exp_target_hemi.right[projection_metric].values),
                                            np.nanmean(exp_target_hemi.both[projection_metric].values)
            )
            for hemi in self.hemispheres_names:
                results[hemi].append(proj_energy._asdict()[hemi])
            results["origin_id"].append(origin)
            results["origin_acronym"].append(origin_acronym)
            results["origin_name"].append(origin_name)

        results = pd.DataFrame.from_dict(results).sort_values("right")
        return results


    def rank_structures(self, SOI): # ! it dont work
        SOI_id = self.structure_tree.get_structures_by_acronym([SOI])[0]["id"]
        experiment_data = pd.read_pickle(os.path.join(self.save_fld, "{}.pkl".format(SOI)))

        ranked = self.mcc.rank_structures(
            experiment_ids = experiment_data.id.values, 
            is_injection  = False, 
            structure_ids = self.structures.id.values, 
            n = 100,
        )

        return ranked



    def plot_structures_rd(self, structures_acronyms):
        # Download OBJ files
        for structure_id in structures_acronyms:
            structure = self.structure_tree.get_structures_by_acronym([structure_id])

            obj_file = os.path.join(self.models_fld, "{}.obj".format(structure[0]["acronym"]))
            if not os.path.isfile(obj_file):
                mesh = self.space.download_structure_mesh(structure_id = structure[0]["id"], ccf_version ="annotation/ccf_2017", 
                                                file_name=obj_file)

        # Create plot
        vp = Plotter(title='first example')

        # plot whole brain
        obj_path = os.path.join(self.models_fld, "root.obj")
        vp.load(obj_path, c=[.8, .8, .8], alpha=.3) 

        
        # plot target brain regions
        for structure in structures_acronyms:
            structure = self.structure_tree.get_structures_by_acronym([structure])
            obj_path = os.path.join(self.models_fld, "{}.obj".format(structure[0]["acronym"]))
            vp.load(obj_path, c=[.8, .3, .3], alpha=1.0) 
        vp.show()  #
        # https://github.com/marcomusy/vtkplotter/tree/master/examples




if __name__ == "__main__":
    analyzer = ConnectivityAnalyzer()
    print("\\n")
    SOI = "PRNc"

    analyzer.plot_structures_rd(["PAG", "SCm"])

    a = 1

