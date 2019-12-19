import napari
import numpy as np
from brainio import brainio

nii_path = "D:\\alignment\\downsampled_2.nii"

with napari.gui_qt():
    v = napari.Viewer(title="amap viewer")

    image_scales = (1, 1, 1)
    image = brainio.load_any(nii_path)
    image = np.swapaxes(image, 2, 0)

    v.add_image(
        image,
        name="Downsampled data",
    )