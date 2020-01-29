

from brainrender.scene import Scene


import pandas as pd

cells = pd.read_hdf(r'Z:\swc\branco\BrainSaw\cellfinder_cells\CC_134_1_test.h5')
cells2 = pd.read_hdf(r'Z:\swc\branco\BrainSaw\cellfinder_cells\CC_134_1_ch1_cells.h5', key='hdf')

scene = Scene()

scene.add_cells(cells, alpha=.5)
scene.add_cells(cells, color='green', alpha=.5)

scene.render()

