# %%
from Utilities.imports import *
%matplotlib inline  
from Modelling.glm.glm_data_loader import GLMdata



# %% Get data
glm = GLMdata(load_trials_from_file=True)
params = glm.load_maze_params()
params.index = params.name

# TODO get arms measurements in CM

# need to convert the lengths from the two expeirments to be on the same scale
conv_fact = 28.368  # ratio of asym.right.len and sym right len in the params df
params["adjusted_length"] = np.divide(params.length.values, conv_fact) # ! this wont be correct for all experiments

params


# %%
# Define utility functions and plotter function

def constant_product(l=None, t=None, x=None, y=None, k=None):
    # indifference curve: x*y = k
    # -> y = k/x
    if l is not None and t is not None:
        return l*t
    elif x is not None and k is not None:
        return k/x
    elif x is not None and y is not None:
        return x.T*y

def constant_sum(l=None, t=None, x=None, y=None, k=None):
    # indifference curve: x+y = k
    # -> y = k-x
    if l is not None and t is not None:
        return l+t
    elif x is not None and k is not None:
        return k-x
    elif x is not None and y is not None:
        return x.T+y


def constant(l=None, t=None, x=None, y=None, k=None):
    # indifference curve: y = kx
    # -> y = k*x
    if l is not None and t is not None:
        return t/l
    elif x is not None and k is not None:
        return x*k
    elif x is not None and y is not None:
        return x*2


# arms = ["asym_right", "asym_left", "sym_right", "sym_left", "mbv2_long", "mbv2_short"]
# lengths, ithetas = [19, 26.085, 19, 19, 45, 28], [135, 180, 135, 135, 270, 180]
# arms_points = {arm:[l, t] for arm, l, t in zip(arms, lengths, ithetas)}
# arms_equilibria = {arm:l*t for arm, l, t in zip(arms, lengths, ithetas)}
# colors = ["red", "red", "m", "m", "orange", "orange"]




def plotter(func, shaded=False, x_max=50):
    # Get vars
    arms = ["asym_right", "asym_left",  "mbv2_long", "mbv2_short"]
    # lengths, ithetas = [19, 26.085,  45, 28], [135, 180,  270, 180]
    # lengths, ithetas = [0.72, 1,   1, 0.622], [135, 180, 270, 180] # ? rLen
    # lengths, ithetas = [19, 26.085,  45, 28], [45, 45,  90, 45] # ? atheta
    lengths, ithetas = [19, 26.085,  45, 28], [180, 225,  360, 225] # ? atheta + iTheta

    arms_points = {arm:[l, t] for arm, l, t in zip(arms, lengths, ithetas)}
    colors = ["red", "red", "orange", "orange"]

    # Get equilibria values
    arms_equilibria = {arm:func(l=l, t=t) for arm, l, t in zip(arms, lengths, ithetas)} # l*t
    equils = sorted(list(arms_equilibria.values()))

    # Get background image
    y = np.tile(np.linspace(0, x_max, 1000), 1000).reshape(1000, 1000)
    x = np.tile(np.linspace(0, x_max, 1000), 1000).reshape(1000, 1000)
    img = func(x=x, y=y)  # x.T*y  # value at all locations

    x = np.linspace(0, x_max, 100)

    f, ax = plt.subplots()
    #  plot indifference curve
    if shaded:
        for k in np.arange(equils[0], equils[1], 10):
            y = func(x=x, k=k)
            ax.plot(x, y, color="red", alpha=.025)

        for k in np.arange(equils[2], equils[-1], 10):
            y = func(x=x, k=k)
            ax.plot(x, y, color="orange", alpha=.025)

    for c, eq in zip(colors, arms_equilibria.values()):
        y = func(x=x, k=eq)
        ax.plot(x, y, color=c)

    # plot vertical and horizontal lines
    for x in lengths:
        ax.axvline(x, linestyle="--", color="w", lw=.15)
    for t in ithetas:
        ax.axhline(t, linestyle="--", color="w", lw=.15)

    # Plot equilibria points
    for c, (k, (xx, yy)) in zip(colors, arms_points.items()):
            ax.plot(xx, yy, 'o', label=k, color=c)


    # show background
    ax.imshow(img, interpolation="nearest", origin="lower", extent=(0,  x_max, 0, 700))

    # set axes
    ax.set(xlabel="arm length", ylabel="iTheta")
    ax.legend()

    return arms_equilibria
#%%
arms_equilibria = plotter(constant_product, x_max=75)

#%%
