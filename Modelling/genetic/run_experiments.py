# %%
import sys
sys.path.append('./')
import itertools


from Utilities.imports import *

from Modelling.genetic.stochastic_vs_efficient import *

%matplotlib inline

# %%
save_fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\plots\\Efficiency_vs_Stochasticity"
# %%
# Base params
# mazes paths lengths
mazes = dict(
	m1 = (1.7, 1.),
	m2 = None,
	m3 = None,
	m4 = (1., 1.)
)

def get_params(maze, **kwargs):
	params = {}
	params["left_l"] = mazes[maze][0]
	params["right_l"] = mazes[maze][1]

	params['danger'] = kwargs.pop('danger', danger)
	params['predator_memory'] = kwargs.pop('predator_memory', predator_memory)
	params['predator_risk_factor'] = kwargs.pop('predator_risk_factor', predator_risk_factor)
	params['reproduction'] = kwargs.pop('reproduction', reproduction)
	params['mutation_rate'] = kwargs.pop('mutation_rate', mutation_rate)
	params['n_agents'] = kwargs.pop('n_agents', n_agents)
	params['max_agents'] = kwargs.pop('max_agents', max_agents)
	params['n_generations'] = kwargs.pop('n_generations', n_generations)
	return params

def restore_params():
	left_l =  1.7
	right_l = 1
	danger = .1 
	predator_memory = 500
	predator_risk_factor = .03 
	reproduction = .5
	mutation_rate = .001
	n_agents = 100
	max_agents = 200 
	n_generations = 600+1

	return left_l, right_l, danger, predator_memory, predator_risk_factor, reproduction, mutation_rate, n_agents, max_agents, n_generations

left_l, right_l, danger, predator_memory, predator_risk_factor, reproduction, mutation_rate, n_agents, max_agents, n_generations = restore_params()

# %%
# Run simulation with base params in asym maze
left_l, right_l, danger, predator_memory, predator_risk_factor, reproduction, mutation_rate, n_agents, max_agents, n_generations = restore_params()

scene = Scene(**get_params("m1"))
scene.run()
ax = scene.plot_summary()
ax.set(facecolor=[.2, .2, .2], xlabel="# generations", ylabel="p(R)", ylim=[0, 1])


# %%
# run maze varying efficiency pressure
left_l, right_l, danger, predator_memory, predator_risk_factor, reproduction, mutation_rate, n_agents, max_agents, n_generations = restore_params()


ch_asym, ch_sym = MplColorHelper("Greens", 0, .5), MplColorHelper("Reds", 0, .5)
dangers = np.arange(0, .25, .05)

fig, axarr = create_figure(subplots=True, nrows=2, facecolor=white, figsize=(16, 16), sharex=True)
for factor in dangers:
	# Run on asymmetric maze
	scene = Scene(**get_params("m1", danger=factor))
	scene.run()
	axarr[0].plot(line_smoother(scene.traces.predator_bias), color=ch_asym.get_rgb(factor), label="EF: {}".format(round(factor,2)))

	# Run on symmetric maze
	scene = Scene(**get_params("m4", danger=factor))
	scene.run()
	axarr[1].plot(line_smoother(scene.traces.predator_bias), color=ch_sym.get_rgb(factor), label="EF: {}".format(round(factor,2)))

axarr[0].axhline(0.5, color=black)
ortholines(axarr[0], [0, 0, 0], [0, 0.5, 1], lw=1, alpha=.8)
axarr[0].legend()
axarr[0].set(title="p(R) vs efficiency pressure. Stochasticity pressure {}. M1".format(round(.03, 2)), 
		facecolor=[.2, .2, .2], xlabel="# generations", ylabel="p(R)", ylim=[-.1, 1.1])

axarr[1].axhline(0.5, color=black)
ortholines(axarr[1], [0, 0, 0], [0, 0.5, 1], lw=1, alpha=.8)
axarr[1].legend()
axarr[1].set(title="p(R) vs efficiency pressure. Stochasticity pressure {}. M4".format(round(.03, 2)), 
		facecolor=[.2, .2, .2], xlabel="# generations", ylabel="p(R)", ylim=[-.1, 1.1])

left_l, right_l, danger, predator_memory, predator_risk_factor, reproduction, mutation_rate, n_agents, max_agents, n_generations = restore_params()


#%%
# run maze varying stochasticity pressure
left_l, right_l, danger, predator_memory, predator_risk_factor, reproduction, mutation_rate, n_agents, max_agents, n_generations = restore_params()

max_v = .1
dangers = np.arange(0, max_v, .0125)
ch_asym, ch_sym = MplColorHelper("Greens", 0, max_v), MplColorHelper("Reds", 0, max_v)

fig, axarr = create_figure(subplots=True, nrows=2, facecolor=white, figsize=(16, 16), sharex=True)
for factor in dangers:
	# Run on asymmetric maze
	scene = Scene(**get_params("m1", predator_risk_factor=factor))
	scene.run()
	axarr[0].plot(line_smoother(scene.traces.predator_bias), color=ch_asym.get_rgb(factor), label="SF: {}".format(round(factor,2)))

	# Run on symmetric maze
	scene = Scene(**get_params("m4", predator_risk_factor=factor))
	scene.run()
	axarr[1].plot(line_smoother(scene.traces.predator_bias), color=ch_sym.get_rgb(factor), label="SF: {}".format(round(factor,2)))

axarr[0].axhline(0.5, color=black)
ortholines(axarr[0], [0, 0, 0], [0, 0.5, 1], lw=1, alpha=.8)
axarr[0].legend()
axarr[0].set(title="p(R) vs stochasticity pressure. Efficiency pressure {}. M1".format(round(.1, 2)), 
		facecolor=[.2, .2, .2], xlabel="# generations", ylabel="p(R)", ylim=[-.1, 1.1])

axarr[1].axhline(0.5, color=black)
ortholines(axarr[1], [0, 0, 0], [0, 0.5, 1], lw=1, alpha=.8)
axarr[1].legend()
axarr[1].set(title="p(R) vs stochasticity pressure. Efficiency pressure {}. M4".format(round(.1, 2)), 
		facecolor=[.2, .2, .2], xlabel="# generations", ylabel="p(R)", ylim=[-.1, 1.1])

left_l, right_l, danger, predator_memory, predator_risk_factor, reproduction, mutation_rate, n_agents, max_agents, n_generations = restore_params()


#%%
# Vary both params
SF_values = [0.0, 0.05, 0.1]
EF_values = [0.0, 0.1, 0.2]
repeats = 3

n_exps = repeats * len(SF_values) * len(EF_values)
curexp = 1

ch = MplColorHelper("Greens", 0, 1)
f, ax = create_figure(subplots=False, facecolor=white, figsize=(16, 16))
pR_values = []
for SF in SF_values: # varying stochasticity pressure
	for EF in EF_values: # varying efficiency pressure
		mean_pR = 0
		for i in range(repeats):
			print("Running {}/{}".format(curexp, n_exps))
			curexp += 1
			scene = Scene(**get_params("m1", predator_risk_factor=SF, risk=EF))
			scene.run()
			pR = np.mean(scene.traces.predator_bias.values[-100:])
			mean_pR += pR
		mean_pR = mean_pR/repeats
		pR_values.append((SF, EF, mean_pR))
		ax.scatter(SF, EF, color=ch.get_rgb(mean_pR), s=1000*mean_pR, label="SF{},EF{},p(R):{}".format(SF, EF, round(mean_pR, 2)))
		

ax.set(title="varying both pressures in M1", xlabel="stochasticity pressure", ylabel="efficiency pressure", facecolor=[.2, .2, .2],)
ax.legend()


#%%
# Fit 2d polinomial to parameters space

def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

# organize Data...
x = np.array([p[0] for p in pR_values])
y = np.array([p[1] for p in pR_values])
z = np.array([p[2] for p in pR_values])

# Fit a 3rd order, 2d polynomial
m = polyfit2d(x,y,z)

# Evaluate it on a grid...
nx, ny = 20, 20
xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx), 
						np.linspace(y.min(), y.max(), ny))
zz = polyval2d(xx, yy, m)

# Plot
f,ax = create_figure(subplots=False, facecolor=white, figsize=(16, 16))
ax.imshow(zz, extent=(0, x.max(), 0, y.max()), cmap="Greens")
ax.scatter(x, y, c=z, cmap="Greens", lw=1, edgecolor=black)

CS = ax.contour(xx, yy, zz)
ax.clabel(CS, inline=1, fontsize=10)
ax.set(xlabel="stochasticity pressure", ylabel="efficiency pressure")


#%%
# Run simulation with selected params
left_l, right_l, danger, predator_memory, predator_risk_factor, reproduction, mutation_rate, n_agents, max_agents, n_generations = restore_params()

scene = Scene(**get_params("m1", danger=0.2, predator_risk_factor=0.05))
scene.run()
ax = scene.plot_summary()
ax.set(facecolor=[.2, .2, .2], xlabel="# generations", ylabel="p(R)", ylim=[0, 1])

#%%
