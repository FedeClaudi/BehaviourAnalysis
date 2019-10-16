# %%
import sys
sys.path.append('./')
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
	left_l =  1.3
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
scene = Scene(**get_params("m1"))
scene.run()
ax = scene.plot_summary()
ax.set(facecolor=[.2, .2, .2], xlabel="# generations", ylabel="p(R)", ylim=[0, 1])


# %%
# run maze varying efficiency pressure
ch_asym, ch_sym = MplColorHelper("Greens", 0, .5), MplColorHelper("Reds", 0, .5)
dangers = np.arange(0, .25, .05)

fig, axarr = create_figure(subplots=True, nrows=2, facecolor=white, figsize=(16, 16), sharex=True)
for danger in dangers:
	# Run on asymmetric maze
	scene = Scene(**get_params("m1", danger=danger))
	scene.run()
	axarr[0].plot(line_smoother(scene.traces.predator_bias), color=ch_asym.get_rgb(danger), label="EF: {}".format(danger))

	# Run on symmetric maze
	scene = Scene(**get_params("m4", danger=danger))
	scene.run()
	axarr[1].plot(line_smoother(scene.traces.predator_bias), color=ch_sym.get_rgb(danger), label="EF: {}".format(danger))

axarr[0].axhline(0.5, color=black)
axarr[0].legend()
axarr[0].set(title="p(R) vs efficiency pressure. Stochasticity pressure {}. M1".format(round(.03, 2)), 
		facecolor=[.2, .2, .2], xlabel="# generations", ylabel="p(R)", ylim=[0, 1])

axarr[1].axhline(0.5, color=black)
axarr[1].legend()
axarr[1].set(title="p(R) vs efficiency pressure. Stochasticity pressure {}. M4".format(round(.03, 2)), 
		facecolor=[.2, .2, .2], xlabel="# generations", ylabel="p(R)", ylim=[0, 1])

left_l, right_l, danger, predator_memory, predator_risk_factor, reproduction, mutation_rate, n_agents, max_agents, n_generations = restore_params()


#%%
# run maze varying stochasticity pressure
ch_asym, ch_sym = MplColorHelper("Greens", 0, .5), MplColorHelper("Reds", 0, .5)
dangers = np.arange(0, .025, .005)

fig, axarr = create_figure(subplots=True, nrows=2, facecolor=white, figsize=(16, 16), sharex=True)
for danger in dangers:
	# Run on asymmetric maze
	scene = Scene(**get_params("m1", predator_risk_factor=danger))
	scene.run()
	axarr[0].plot(line_smoother(scene.traces.predator_bias), color=ch_asym.get_rgb(danger), label="SF: {}".format(danger))

	# Run on symmetric maze
	scene = Scene(**get_params("m4", predator_risk_factor=danger))
	scene.run()
	axarr[1].plot(line_smoother(scene.traces.predator_bias), color=ch_sym.get_rgb(danger), label="SF: {}".format(danger))

axarr[0].axhline(0.5, color=black)
axarr[0].legend()
axarr[0].set(title="p(R) vs efficiency pressure. Efficiency pressure {}. M1".format(round(.1, 2)), 
		facecolor=[.2, .2, .2], xlabel="# generations", ylabel="p(R)", ylim=[0, 1])

axarr[1].axhline(0.5, color=black)
axarr[1].legend()
axarr[1].set(title="p(R) vs efficiency pressure. Efficiency pressure {}. M4".format(round(.1, 2)), 
		facecolor=[.2, .2, .2], xlabel="# generations", ylabel="p(R)", ylim=[0, 1])

left_l, right_l, danger, predator_memory, predator_risk_factor, reproduction, mutation_rate, n_agents, max_agents, n_generations = restore_params()


#%%
