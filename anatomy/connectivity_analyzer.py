import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import namedtuple
from vtkplotter import Plotter, show, interactive, Video, settings

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.api.queries.ontologies_api import OntologiesApi
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi

from anatomy.mouselight_parser import render_neurons

class ConnectivityAnalyzer:
    main_fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\anatomy"
    fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\anatomy\\mouse_connectivity"
    models_fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\anatomy\\3dModels"
    neurons_fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\anatomy\\Mouse Light"

    hemispheres = namedtuple("hemispheres", "left right both") # left: CONTRA, right: IPSI
    hemispheres_names = ["left", "right", "both"]
    
    # useful vars for analysis
    projection_metric = "projection_energy"
    volume_threshold = 0.5
    excluded_regions = ["fiber tracts"]

    def __init__(self):
        manifest = os.path.join(self.fld, "manifest.json")
        self.mcc = MouseConnectivityCache(manifest_file=manifest)

        self.structure_tree = self.mcc.get_structure_tree()
        
        # Get the main structures names, ID...
        oapi = OntologiesApi()
        summary_structures = self.structure_tree.get_structures_by_set_id([167587189])
        summary_structures = [s for s in summary_structures if s["acronym"] not in self.excluded_regions]
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


    def analyze_efferents(self, SOI, projection_metric = None):
        """[Loads the experiments on SOI and looks at average statistics of efferent projections]
        
        Arguments:
            SOI {[str]} -- [acronym of the structure of interest to look at]
        """
        if projection_metric is None: 
            projection_metric = self.projection_metric

        experiment_data = pd.read_pickle(os.path.join(self.save_fld, "{}.pkl".format(SOI)))
        experiment_data = experiment_data.loc[experiment_data.volume > self.volume_threshold]

        # Loop over all structures and get the injection density
        results = {"left":[], "right":[], "both":[], "id":[], "acronym":[], "name":[]}
        for target in self.structures.id.values:
            target_acronym = self.structures.loc[self.structures.id == target].acronym.values[0]
            target_name = self.structures.loc[self.structures.id == target].name.values[0]

            exp_target = experiment_data.loc[experiment_data.structure_id == target]

            exp_target_hemi = self.hemispheres(exp_target.loc[exp_target.hemisphere_id == 1], exp_target.loc[exp_target.hemisphere_id == 2], exp_target.loc[exp_target.hemisphere_id == 3])
            proj_energy = self.hemispheres(np.nanmean(exp_target_hemi.left[projection_metric].values),
                                            np.nanmean(exp_target_hemi.right[projection_metric].values),
                                            np.nanmean(exp_target_hemi.both[projection_metric].values)
            )


            for hemi in self.hemispheres_names:
                results[hemi].append(proj_energy._asdict()[hemi])
            results["id"].append(target)
            results["acronym"].append(target_acronym)
            results["name"].append(target_name)

        results = pd.DataFrame.from_dict(results).sort_values("right", na_position = "first")
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
        results = {"left":[], "right":[], "both":[], "id":[], "acronym":[], "name":[]}

        for origin in self.structures.id.values:
            origin_acronym = self.structures.loc[self.structures.id == origin].acronym.values[0]
            origin_name = self.structures.loc[self.structures.id == origin].name.values[0]

            experiment_data = pd.read_pickle(os.path.join(self.save_fld, "{}.pkl".format(origin_acronym)))
            experiment_data = experiment_data.loc[experiment_data.volume > self.volume_threshold]

            exp_target = experiment_data.loc[experiment_data.structure_id == SOI_id]
            exp_target_hemi = self.hemispheres(exp_target.loc[exp_target.hemisphere_id == 1], exp_target.loc[exp_target.hemisphere_id == 2], exp_target.loc[exp_target.hemisphere_id == 3])
            proj_energy = self.hemispheres(np.nanmean(exp_target_hemi.left[projection_metric].values),
                                            np.nanmean(exp_target_hemi.right[projection_metric].values),
                                            np.nanmean(exp_target_hemi.both[projection_metric].values)
            )
            for hemi in self.hemispheres_names:
                results[hemi].append(proj_energy._asdict()[hemi])
            results["id"].append(origin)
            results["acronym"].append(origin_acronym)
            results["name"].append(origin_name)

        results = pd.DataFrame.from_dict(results).sort_values("right", na_position = "first")
        return results



    def plot_structures_3d(self, structures_acronyms, default_colors=True, verbose=False, target=None, target_color=[.4, .4, .4], others_color=[.4, .4, .4],
                        sagittal_slice=False, neurons_file=None, render=True, neurons_kwargs={}, specials=[]):
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
        root = vp.load(obj_path, c=[.8, .8, .8], alpha=.3)  


        # Plot target structure
        if target is not None:
            obj_path = os.path.join(self.models_fld, "{}.obj".format(target))
            if not os.path.isfile(obj_file):
                mesh = self.space.download_structure_mesh(structure_id = structure[0]["id"], ccf_version ="annotation/ccf_2017", 
                                                file_name=obj_file)

            target_mesh = vp.load(obj_path, c=target_color, alpha=1)

 
        # plot other brain regions
        other_structures = []
        for i, structure in enumerate(structures_acronyms):
            if target is not None:
                if structure == target: continue
            structure = self.structure_tree.get_structures_by_acronym([structure])

            if default_colors:
                color = [x/255 for x in structure[0]["rgb_triplet"]]
            else: 
                if isinstance(others_color[0], list):
                    color = others_color[i]
                else:
                    color = others_color

            if structure[0]["acronym"] in specials:
                color = "steelblue"
                alpha = 1
            else:
                alpha = 0.1

            obj_path = os.path.join(self.models_fld, "{}.obj".format(structure[0]["acronym"]))
            mesh = vp.load(obj_path, c=color, alpha=alpha) 
            other_structures.append(mesh)

            if verbose:
                print("Rendering: {} - ({})".format(structure[0]["name"], structure[0]["acronym"]))

        if sagittal_slice:
            for a in vp.actors:
                a.cutWithPlane(origin=(0,0,6000), normal=(0,0,-1), showcut=True)


        # add neurons
        if neurons_file is not None:
            neurons_actors = render_neurons(neurons_file, **neurons_kwargs)
        else:
            neurons_actors = []

        # Add sliders
        if target is not None:
            def target_slider(widget, event):
                value = widget.GetRepresentation().GetValue()
                target_mesh.alpha(value)

            vp.addSlider2D(target_slider, xmin=0.01, xmax=0.99, value=0.5, pos=4, title="target alpha")

        def others_slider(widget, event):
            value = widget.GetRepresentation().GetValue()
            for actor in other_structures:
                actor.alpha(value)
        vp.addSlider2D(others_slider, xmin=0.01, xmax=0.99, value=0.5, pos=3, title="others alpha")

        # Add inset 
        inset = root.clone().scale(.5)
        inset.alpha(1)
        vp.showInset(inset, pos=(0.9,0.2))

        if render:
            show(*vp.actors, *neurons_actors, interactive=True, roll=180, azimuth=-35, elevation=-25)  
        else:
            show(*vp.actors, *neurons_actors, interactive=0, offscreen=True, roll=180)  
            return vp


    def video_maker(self, dest_path, *args, **kwargs):
        vp = self.plot_structures_3d(*args, render=False, **kwargs)

        fld, video = os.path.split(dest_path)
        os.chdir(fld)
        video = Video(name=video, duration=3)
        
        for i  in range(80):
            vp.show()  # render the scene first
            vp.camera.Elevation(5)  # rotate by 5 deg at each iteration
            # vp.camera.Zoom(i/40)
            video.addFrame()
        video.close()  # merge all the recorded frames





if __name__ == "__main__":
    # rendere settings
    settings.useOpenVR = False



    analyzer = ConnectivityAnalyzer()

    SOI = "GRN"
    neurons = os.path.join(analyzer.neurons_fld, "axons_in_Gi.json")
    efferents = analyzer.analyze_efferents(SOI, projection_metric="normalized_projection_volume")
    analyzer.plot_structures_3d(efferents.acronym.values[-30:], verbose=True, sagittal_slice=False,
                                    default_colors=False,
                                    target = SOI,
                                    target_color="red",
                                    others_color="palegoldenrod",
                                    neurons_file = None,
                                    specials=["PAG", "SCm", "ZI"])


    # videopath = os.path.join(analyzer.main_fld, "GRN.mp4")
    # analyzer.video_maker(videopath, efferents.acronym.values[-30:], verbose=True, sagittal_slice=False,
    #                                 default_colors=False,
    #                                 target = None,
    #                                 target_color="red",
    #                                 others_color="palegoldenrod",
    #                                 neurons_file = None, neurons_kwargs=dict(neurite_radius=25),
    #                                 specials=["PAG", "SCm", "ZI"])


    # print(analyzer.structures)
    # [print(x.acronym) for i,x in analyzer.structures.iterrows() if "superior" in x["name"].lower()]