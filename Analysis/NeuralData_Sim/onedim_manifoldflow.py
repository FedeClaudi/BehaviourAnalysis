
# %%
import sys
sys.path.append("./")
from Utilities.imports import *
from random import random
# %matplotlib inline



# %%
# parameters
n_steps = 5000

latent_var_params = dict(
    stepsize = 10,
    drift = 0, 
    b0 = 1, # weight of previous step
    noise_std = 5
)

# %%
# Neuron class definition
class Neuron:
    def __init__(self, tuning_curve_func, noise_std=.1):
        self.tuning_curve = tuning_curve_func
        self.noise_std = noise_std

        self.history = []
    
    def get_rate(self, latent_var):
        cur_far = round(self.tuning_curve(latent_var) + np.random.normal(0, self.noise_std), 2)
        self.history.append(cur_far)
        return cur_far

def n1_tuning_curve(latent_var):
    return np.sin(np.radians(latent_var)) + 1.5

def n2_tuning_curve(latent_var):
    return np.cos(np.radians(latent_var)) + 1.5

# %%
# Random walk functions

def step_latent_variable(walk, params):
    prev = walk[-1]
    if random() > .5:
        step = params['stepsize']
    else:
        step = - params['stepsize']
    

    # y(t) = B0 + B1*X(t-1) + e(t)
    nxt = params['drift'] + walk[-1] + np.random.normal(0, params['noise_std'])

    walk.append(nxt)
    return walk





# %%
# walk the walk, plot the plots
walk = [180]

neurons = [Neuron(n1_tuning_curve), Neuron(n2_tuning_curve)]

for t in range(n_steps):
    # step latent variable
    walk = step_latent_variable(walk, latent_var_params)

    # get neurons firing rates
    for neuron in neurons:
        neuron.get_rate(walk[-1])

# Create population history
popvect = np.vstack([n.history for n in neurons])
popvect_shifted = popvect[:, 1:]

# Plot summary
f, axarr = create_figure(subplots=True, ncols=2, nrows=2, figsize=(16,8))

axarr[0].plot(walk, color=black)
axarr[1].scatter(neurons[0].history, neurons[1].history, c=np.arange(len(neurons[0].history)),
                cmap="Reds")

_ = axarr[2].plot([popvect[0, 0:-1:4], popvect_shifted[0, 0:-1:4]],
            [popvect[1, 0:-1:4], popvect_shifted[1, 0:-1:4]])

_ = axarr[0].set(title="Latent variable")
_ = axarr[1].set(title="pop vector over time", xlim=[0, 3], ylim=[0, 3])
_ = axarr[2].set(title="Pop vector 'steps'", xlim=[0, 3], ylim=[0, 3])





# %%
# Get dynamics

# Get a list of all the places the pop has been
poplist = [(x,y) for x,y in zip(*popvect.tolist())]
places = set(poplist)

# get avg vector at each location
vectors = []
for place in tqdm(places):
    idxs = [i for i,p in enumerate(poplist) if p == place]
    
    place_vectors = [[], []]
    for idx in idxs:
        if idx+1 == len(poplist): continue
        place_vectors[0].append(poplist[idx+1][0]-poplist[idx][0])
        place_vectors[1].append(poplist[idx+1][1]-poplist[idx][1])

    vectors.append((np.mean(place_vectors[0]), np.mean(place_vectors[1])))

vectors_in_place = [(v[0]+p[0], v[1]+p[1]) for v,p in zip(vectors, places)] # for plotting
vectors_lengths = [(math.sqrt((v[0]**2 + v[1]**2))) for v in vectors_in_place]

# %%
# plot vector field

ch = MplColorHelper("Reds", 0, max(vectors_lengths))
colors = [ch.get_rgb(l) for l in vectors_lengths]
f, axarr = create_figure(subplots=True, ncols=2)

_ = axarr[0].hist(vectors_lengths, bins=25, color=blue)
_ = sns.plot([[p[0] for p in places], [v[0] for v in vectors_in_place]],
        [[p[1] for p in places], [v[1] for v in vectors_in_place]],
        color= colors, ax=axarr[1]
        )


axarr[0].set(title="vectors lengths")
axarr[1].set(title="title pop vectors")

# %%

# %%


# %%
