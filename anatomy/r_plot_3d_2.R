library(cocoframer)
library(purrr)
library(rgl)


# https://github.com/AllenInstitute/cocoframer

structures <- c("root", 
		'ZI', 'PAG', 'AHN', 'LM', 'CUN', 'SCO', 'MEV', 'PBG', 'SAG', 'VMH'
)



mesh_list <- map(structures, ccf_2017_mesh)
names(mesh_list) <- structures


plot_ccf_meshes(mesh_list,
                fg_structure = c( 
		'ZI', 'PAG', 'AHN', 'LM', 'CUN', 'SCO', 'MEV', 'PBG', 'SAG', 'VMH'
			),
		    fg_alpha = 0.75,
                bg_structure = "root",
                style = "matte")


