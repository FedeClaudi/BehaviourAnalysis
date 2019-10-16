# %%
import sys
sys.path.append('./')
from Utilities.imports import *

from .stochastic_vs_efficient import *

# %%
# Base params
left_l =  1.3
right_l = 1
danger = .1 
predator_memory = 800
predator_risk_factor = .04 
reproduction = .3  # ! different!!
mutation_rate = .001
n_agents = 100
max_agents = 200 
n_generations = 600+1

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
    params['left_l '] = kwargs.pop('left_l ', left_l )
    params['right_l'] = kwargs.pop('right_l', right_l)
    params['danger'] = kwargs.pop('danger', danger)
    params['predator_memory'] = kwargs.pop('predator_memory', predator_memory)
    params['predator_risk_factor'] = kwargs.pop('predator_risk_factor', predator_risk_factor)
    params['reproduction'] = kwargs.pop('reproduction', reproduction)
    params['mutation_rate'] = kwargs.pop('mutation_rate', mutation_rate)
    params['n_agents'] = kwargs.pop('n_agents', n_agents)
    params['max_agents'] = kwargs.pop('max_agents', max_agents)
    params['n_generations'] = kwargs.pop('n_generations', n_generations)
    return params



# %%
# run maze vary efficiency pressure
dangers = np.arange(0, 1, .2)
for danger in danges:
    


