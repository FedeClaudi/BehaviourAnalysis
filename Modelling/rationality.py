"""
Given the mice's distribution of escape speeds and escape path lengths on the 2 arms experiments>
For a range of speeds calculate the value of each path -> expected duration
Then multiply these values by their probability

Then simulate N trials in which you draw two random speed, one per arm, and compare the values you'd expect



"""

# %%
import sys
sys.path.append("./")
from Utilities.imports import *
from Modelling.glm.glm_data_loader import GLMdata


data = GLMdata

#%%
