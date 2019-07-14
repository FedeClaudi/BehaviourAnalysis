# %%
import math
import numpy as np
from Processing.plot.plot_distributions import plot_distribution

# %maptlotlib inline

# %%
ax = plot_distribution(5, 5, dist_type="beta", color="r")
ax = plot_distribution(14, 6, dist_type="beta", color="g", ax=ax, ylim=[0,4.5], xlim=[0,1])

#%%
