# %%
import sys
sys.path.append("./")

from Utilities.imports import *
# %matplotlib inline


# %%
# load and visualize maze
fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\maze_solvers\\mazes_images"
model = cv2.imread(os.path.join(fld, "PathInt2_old.png"))
model = cv2.blur(model, (25, 25))

# resize and change color space
model = cv2.resize(model, (1000, 1000))
model = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)

# threshold and rotate  and flip on y axis
ret, model = cv2.threshold(model, 50, 255, cv2.THRESH_BINARY)
model = np.rot90(model, 2)
model = model[:, ::-1]

# %%
# Load rois locations
rois = load_yaml("database\maze_components\MazeModelROILocation.yml")


# %%
f, ax = create_figure(subplots=False, figsize=(16, 16)) 
ax.imshow(model)

for name, loc in rois.items():
    if "p" in name or "s" in name or "t" in name: continue 
    c = red
    s = 100
    ax.scatter(loc[0], loc[1], s=s, label=name)

ax.legend()


# %%
print(MazeComponents())

# %%
