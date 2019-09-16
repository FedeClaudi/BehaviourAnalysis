library(cocoframer)
library(purrr)
library(rgl)

structures <- c("root","PAG","SCm","ZI","RE","CA1", "PL", "MOs")
mesh_list <- map(structures, ccf_2017_mesh)
names(mesh_list) <- structures


plot_ccf_meshes(mesh_list,
                fg_structure = c("PAG","SCm","ZI","RE","CA1", "PL", "MOs"),
		    fg_color = c("cornflowerblue","coral2","chartreuse3","aquamarine3","aquamarine4","aquamarine4","aquamarine4"),
                bg_structure = "root",
                style = "matte")



# anim <- spin3d(axis=c(0,1,0), # Spin on the y-axis
               rpm = 12)

# movie3d(anim, 
#        fps = 20, # 20 fps is fairly smooth
#        duration = 5, # 5 sec = 1 rotation at 12 rpm
#        movie = "brain_demo", # Save as brain_demo.gif
#        dir = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\anatomy\\plots\\3d", # Output directory - will make a lot of temporary files.
#        type = "gif")