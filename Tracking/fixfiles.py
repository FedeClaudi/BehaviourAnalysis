# %%
import sys
sys.path.append('./')

import os
from tqdm import tqdm
import shutil


# %%
# Get filtered new pose data
newpose_fld = "Z:\\branco\\Federico\\raw_behaviour\\maze\\newpose"
filtered_newpose = [f for f in os.listdir(newpose_fld) if "filtered" in f and ".h5" in f]
len(filtered_newpose)
# %%
pose_fld = "Z:\\branco\\Federico\\raw_behaviour\\maze\\pose"
for origin in tqdm(filtered_newpose):
    corigin = os.path.join(newpose_fld, origin)

    cleaned = origin.split("DLC")[0]
    if not len(cleaned.split("_"))>2:
        cleaned = cleaned +  "_1"

    dest =cleaned +"_pose.h5"
    cdeset = os.path.join(pose_fld, dest)

    if os.path.isfile(cdeset):
        continue
    shutil.copy(corigin, cdeset)

# %%
os.chdir(pose_fld)
for f in os.listdir(pose_fld):
    new = f.split("_pose")[0]+"Overview_pose.h5"
    shutil.copy(f, new)

# %%
