import itertools
import numpy as np

mazes = dict(
	m1 = (1.39, 1.),
	m2 = (1.18, 1.),
	m3 = (1.1, 1.),
	m4 = (1., 1.)
)

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
	n_generations = 100+1

	return left_l, right_l, danger, predator_memory, predator_risk_factor, reproduction, mutation_rate, n_agents, max_agents, n_generations


def get_params(maze, **kwargs):
	left_l, right_l, danger, predator_memory, predator_risk_factor, reproduction, mutation_rate, n_agents, max_agents, n_generations = restore_params()
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
